#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace {
struct Geometry { unsigned m, n, k, x, y, simdM, simdN; };
constexpr Geometry GEOMETRY[] = {
    {16, 16, 1, 16, 16, 0, 0}, {32, 32, 16, 16, 16, 0, 0},
    {64, 64, 16, 128, 1, 32, 32}, {},
    {64, 64, 0, 128, 1, 0, 0}, {64, 64, 0, 128, 1, 0, 0},
};
constexpr unsigned MORTON_SIDE = 8;
struct Dimensions { uint32_t m, n, k; };
struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipelines[6][3];
    NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *multiplications;
    int language;
    bool tensorOps, neuralAccelerators;
};
struct Plan {
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> a, b, c;
    MPSMatrix *ma, *mb, *mc;
    MPSMatrixMultiplication *mps;
    NSString *failure;
    NSMutableArray<id<MTLCommandBuffer>> *pending;
    Dimensions dims;
    MTLSize grid, threads;
    size_t strideA, strideB, strideC;
    int implementation, inputType;
    bool morton;
};
int fail(char *error, size_t size, NSString *message) {
    if (error && size) snprintf(error, size, "%s", message ? message.UTF8String : "unknown Metal error");
    return -1;
}
int exception_error(char *error, size_t size, NSException *exception) {
    return fail(error, size, [NSString stringWithFormat:@"%@: %@", exception.name, exception.reason]);
}
int copy_json(NSDictionary *value, char *out, size_t size, char *error, size_t errorSize) {
    NSError *err = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:value options:0 error:&err];
    if (!data) return fail(error, errorSize, err.localizedDescription);
    if (!out || data.length + 1 > size) return fail(error, errorSize, @"JSON output buffer is too small");
    memcpy(out, data.bytes, data.length); out[data.length] = 0;
    return 0;
}
bool product(size_t a, size_t b, size_t *out) {
    if (a && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b; return true;
}
int sync_plan(Plan *p, char *error, size_t size) {
    for (id<MTLCommandBuffer> command in p->pending) {
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            NSString *detail = command.error.localizedDescription ?: @"Metal command buffer did not complete successfully";
            p->failure = p->failure ? [p->failure stringByAppendingFormat:@"; %@", detail] : detail;
        }
    }
    [p->pending removeAllObjects];
    return p->failure ? fail(error, size, p->failure) : 0;
}
template<int Type>
void copy_input16(void *destination, const void *source, size_t rows, size_t cols, size_t stride) {
    if (stride == cols * 2) { cols *= rows; rows = 1; }
    for (size_t row = 0; row < rows; ++row) {
        void *dst = static_cast<char *>(destination) + row * stride;
        const char *src = static_cast<const char *>(source) + row * cols * sizeof(float);
        for (size_t i = 0; i < cols; ++i) {
            // memcpy also permits byte-unaligned, contiguous NumPy inputs.
            float value;
            memcpy(&value, src + i * sizeof(float), sizeof(value));
            if constexpr (Type == 1) static_cast<__fp16 *>(dst)[i] = static_cast<__fp16>(value);
            else {
                uint32_t bits;
                memcpy(&bits, &value, sizeof(bits));
                uint16_t rounded;
                if ((bits & 0x7f800000u) == 0x7f800000u)
                    rounded = uint16_t(bits >> 16) | ((bits & 0x7fffffu) ? 0x40u : 0u);
                else rounded = uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
                static_cast<uint16_t *>(dst)[i] = rounded;
            }
        }
    }
}
void copy_input(id<MTLBuffer> buffer, const void *source, size_t rows, size_t cols, size_t stride, int type) {
    void *destination = buffer.contents;
    if (type == 1) copy_input16<1>(destination, source, rows, cols, stride);
    else if (type == 2) copy_input16<2>(destination, source, rows, cols, stride);
    else if (stride == cols * sizeof(float)) memcpy(destination, source, rows * stride);
    else for (size_t row = 0; row < rows; ++row)
        memcpy(static_cast<char *>(destination) + row * stride,
               static_cast<const char *>(source) + row * cols * sizeof(float), cols * sizeof(float));
}
int encode(Plan *p, id<MTLCommandBuffer> command, int count, char *error, size_t size) {
    for (int i = 0; i < count; ++i) {
        if (p->implementation == 3) {
            [p->mps encodeToCommandBuffer:command leftMatrix:p->ma rightMatrix:p->mb resultMatrix:p->mc];
        } else {
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) return fail(error, size, @"could not create Metal compute encoder");
            [encoder setComputePipelineState:p->pipeline];
            [encoder setBuffer:p->a offset:0 atIndex:0];
            [encoder setBuffer:p->b offset:0 atIndex:1];
            [encoder setBuffer:p->c offset:0 atIndex:2];
            [encoder setBytes:&p->dims length:sizeof(p->dims) atIndex:3];
            [encoder dispatchThreadgroups:p->grid threadsPerThreadgroup:p->threads];
            [encoder endEncoding];
        }
    }
    return 0;
}
}

extern "C" {
int metal_abi_version() { return 1; }
void *metal_create(int index, const char *sourcePath, char *error, size_t errorSize) {
    @autoreleasepool { @try {
        if (index < 0 || !sourcePath) { fail(error, errorSize, @"invalid device index or shader path"); return nullptr; }
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (NSUInteger(index) >= devices.count) { fail(error, errorSize, @"Metal device index is unavailable"); return nullptr; }
        std::unique_ptr<Context> ctx(new (std::nothrow) Context());
        if (!ctx) { fail(error, errorSize, @"out of memory creating Metal context"); return nullptr; }
        ctx->device = devices[index];
        ctx->queue = [ctx->device newCommandQueue];
        ctx->multiplications = [NSMutableDictionary new];
        NSError *err = nil;
        NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:sourcePath] encoding:NSUTF8StringEncoding error:&err];
        if (!source) { fail(error, errorSize, err.localizedDescription); return nullptr; }
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        ctx->language = 310;
        options.languageVersion = MTLLanguageVersion3_1;
        if (@available(macOS 26.0, *)) {
            ctx->language = 400;
            ctx->tensorOps = true;
            ctx->neuralAccelerators = [ctx->device supportsFamily:MTLGPUFamilyApple10];
            options.languageVersion = MTLLanguageVersion4_0;
        }
        // One source of truth for shader geometry, dispatch and reported metadata.
        const Geometry &t = GEOMETRY[1], &s = GEOMETRY[2], &v = GEOMETRY[4];
        options.preprocessorMacros = @{
            @"TILED_M": @(t.m), @"TILED_N": @(t.n), @"TILED_K": @(t.k), @"TILED_X": @(t.x), @"TILED_Y": @(t.y),
            @"SIMD_M": @(s.m), @"SIMD_N": @(s.n), @"SIMD_K": @(s.k), @"SIMD_THREADS": @(s.x),
            @"SIMD_WM": @(s.simdM), @"SIMD_WN": @(s.simdN),
            @"TENSOR_M": @(v.m), @"TENSOR_N": @(v.n), @"TENSOR_SIMDGROUPS": @(v.x / 32), @"MORTON_SIDE": @(MORTON_SIDE)};
        ctx->library = [ctx->device newLibraryWithSource:source options:options error:&err];
        if (!ctx->queue || !ctx->library) { fail(error, errorSize, err.localizedDescription ?: @"could not create Metal command queue or library"); return nullptr; }
        return ctx.release();
    } @catch (NSException *ex) { exception_error(error, errorSize, ex); return nullptr; } }
}
void metal_destroy(void *context) { delete static_cast<Context *>(context); }
int metal_info(void *context, char *out, size_t outSize, char *error, size_t errorSize) {
    @autoreleasepool { @try {
        Context *c = static_cast<Context *>(context); if (!c) return fail(error, errorSize, @"closed context");
        MTLSize max = c->device.maxThreadsPerThreadgroup;
        return copy_json(@{@"name": c->device.name, @"unified_memory": @(c->device.hasUnifiedMemory),
                           @"max_threads_per_threadgroup": @[@(max.width), @(max.height), @(max.depth)],
                           @"max_threadgroup_memory_bytes": @(c->device.maxThreadgroupMemoryLength),
                           @"recommended_max_working_set_bytes": @(c->device.recommendedMaxWorkingSetSize),
                           @"metal_language_version": @(c->language), @"tensor_ops": @(c->tensorOps),
                           @"m5_neural_accelerators": @(c->neuralAccelerators)}, out, outSize, error, errorSize);
    } @catch (NSException *ex) { return exception_error(error, errorSize, ex); } }
}
void *metal_prepare(void *context, int impl, int inputType, const void *a, const void *b,
                    int m, int n, int k, char *error, size_t errorSize) {
    @autoreleasepool { @try {
        Context *ctx = static_cast<Context *>(context);
        if (!ctx || !a || !b || m <= 0 || n <= 0 || k <= 0 || impl < 0 || impl > 5 || inputType < 0 || inputType > 2 ||
            (impl < 4 && inputType == 2)) {
            fail(error, errorSize, @"invalid plan arguments"); return nullptr;
        }
        if (impl >= 4 && !ctx->tensorOps) { fail(error, errorSize, @"TensorOps require macOS 26 and Metal 4"); return nullptr; }
        std::unique_ptr<Plan> p(new (std::nothrow) Plan());
        if (!p) { fail(error, errorSize, @"out of memory creating plan"); return nullptr; }
        p->queue = ctx->queue; p->implementation = impl; p->inputType = inputType;
        p->dims = {uint32_t(m), uint32_t(n), uint32_t(k)}; p->pending = [NSMutableArray new];
        const Geometry &g = GEOMETRY[impl];
        if (impl != 3) {
            size_t nx = (size_t(n) + g.n - 1) / g.n, ny = (size_t(m) + g.m - 1) / g.m;
            p->morton = impl == 5 && nx >= MORTON_SIDE && ny >= MORTON_SIDE;
            if (impl == 5) {
                size_t groups = p->morton ? ((nx + MORTON_SIDE - 1) / MORTON_SIDE) * ((ny + MORTON_SIDE - 1) / MORTON_SIDE) * MORTON_SIDE * MORTON_SIDE : nx * ny;
                p->grid = MTLSizeMake(groups, 1, 1);
            } else p->grid = MTLSizeMake(nx, ny, 1);
            p->threads = MTLSizeMake(g.x, g.y, 1);
        }
        size_t element = inputType ? 2 : 4;
        p->strideA = size_t(k) * element; p->strideB = size_t(n) * element; p->strideC = size_t(n) * 4;
        MPSDataType inputDataType = inputType ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        if (impl == 3) {
            p->strideA = [MPSMatrixDescriptor rowBytesForColumns:k dataType:inputDataType];
            p->strideB = [MPSMatrixDescriptor rowBytesForColumns:n dataType:inputDataType];
            p->strideC = [MPSMatrixDescriptor rowBytesForColumns:n dataType:MPSDataTypeFloat32];
        }
        size_t bytesA, bytesB, bytesC;
        if (!product(p->strideA, m, &bytesA) || !product(p->strideB, k, &bytesB) || !product(p->strideC, m, &bytesC)) {
            fail(error, errorSize, @"matrix allocation size overflow"); return nullptr;
        }
        p->a = [ctx->device newBufferWithLength:bytesA options:MTLResourceStorageModeShared];
        p->b = [ctx->device newBufferWithLength:bytesB options:MTLResourceStorageModeShared];
        p->c = [ctx->device newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!p->a || !p->b || !p->c) { fail(error, errorSize, @"could not allocate shared Metal matrix buffers"); return nullptr; }
        copy_input(p->a, a, m, k, p->strideA, inputType);
        copy_input(p->b, b, k, n, p->strideB, inputType);
        if (impl == 3) {
            MPSMatrixDescriptor *da = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:p->strideA dataType:inputDataType];
            MPSMatrixDescriptor *db = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:p->strideB dataType:inputDataType];
            MPSMatrixDescriptor *dc = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:p->strideC dataType:MPSDataTypeFloat32];
            p->ma = [[MPSMatrix alloc] initWithBuffer:p->a descriptor:da];
            p->mb = [[MPSMatrix alloc] initWithBuffer:p->b descriptor:db];
            p->mc = [[MPSMatrix alloc] initWithBuffer:p->c descriptor:dc];
            NSString *key = [NSString stringWithFormat:@"%d/%d/%d", m, k, n];
            p->mps = ctx->multiplications[key];
            if (!p->mps) {
                p->mps = [[MPSMatrixMultiplication alloc] initWithDevice:ctx->device transposeLeft:NO transposeRight:NO resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
                p->mps.options = MPSKernelOptionsNone;
                if (p->mps) ctx->multiplications[key] = p->mps;
            }
            if (!p->ma || !p->mb || !p->mc || !p->mps) { fail(error, errorSize, @"could not create MPS matrices or multiplication"); return nullptr; }
        } else {
            p->pipeline = ctx->pipelines[impl][inputType];
            if (!p->pipeline) {
                NSString *name = [NSString stringWithFormat:@"%@_%@", (@[@"naive", @"tiled", @"simd", @"", @"tensor", @"tensor_morton"])[impl], (@[@"f32", @"f16", @"bf16"])[inputType]];
                id<MTLFunction> function = [ctx->library newFunctionWithName:name];
                NSError *err = nil;
                if (function) {
                    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
                    descriptor.computeFunction = function;
                    descriptor.maxTotalThreadsPerThreadgroup = g.x * g.y;
                    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                    p->pipeline = [ctx->device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&err];
                }
                if (!p->pipeline) { fail(error, errorSize, err.localizedDescription ?: [@"missing Metal function " stringByAppendingString:name]); return nullptr; }
                ctx->pipelines[impl][inputType] = p->pipeline;
            }
            if (p->pipeline.maxTotalThreadsPerThreadgroup < g.x * g.y || (impl >= 2 && p->pipeline.threadExecutionWidth != 32) || p->pipeline.staticThreadgroupMemoryLength > ctx->device.maxThreadgroupMemoryLength) {
                fail(error, errorSize, @"device does not support the selected Metal kernel geometry"); return nullptr;
            }
        }
        return p.release();
    } @catch (NSException *ex) { exception_error(error, errorSize, ex); return nullptr; } }
}
int metal_config(void *plan, char *out, size_t outSize, char *error, size_t errorSize) {
    @autoreleasepool { @try {
        Plan *p = static_cast<Plan *>(plan); if (!p) return fail(error, errorSize, @"closed plan");
        const Geometry &g = GEOMETRY[p->implementation];
        NSArray *tile = p->implementation == 3 ? @[] : @[@(g.m), @(g.n), g.k ? @(g.k) : @"dynamic"];
        NSDictionary *config = @{@"implementation": (@[@"metal_naive", @"metal_tiled", @"metal_simd", @"mps", @"metal_tensor", @"metal_tensor_morton"])[p->implementation],
            @"input_dtype": (@[@"float32", @"float16", @"bfloat16"])[p->inputType],
            @"accumulation_dtype": @"float32", @"output_dtype": @"float32", @"tile_mnk": tile,
            @"threads_per_threadgroup": p->implementation == 3 ? @[] : @[@(g.x), @(g.y), @1],
            @"grid_threadgroups": p->implementation == 3 ? @[] : @[@(p->grid.width), @(p->grid.height), @(p->grid.depth)],
            @"grid_order": p->implementation == 3 ? @"opaque_vendor" : p->morton ? [NSString stringWithFormat:@"blocked_morton_%ux%u", MORTON_SIDE, MORTON_SIDE] : @"row_major",
            @"tensor_primitive": @(p->implementation >= 4), @"relaxed_precision": @NO,
            @"row_stride_bytes": @[@(p->strideA), @(p->strideB), @(p->strideC)],
            @"math_mode": p->implementation == 3 ? @"mps_no_reduced_precision" : @"safe_precise",
            @"thread_execution_width": p->pipeline ? @(p->pipeline.threadExecutionWidth) : (id)[NSNull null],
            @"threadgroup_memory_bytes": p->pipeline ? @(p->pipeline.staticThreadgroupMemoryLength) : (id)[NSNull null],
            @"timing": @"command_buffer_gpu_timestamps", @"batching": @"dispatches_per_command_buffer"};
        return copy_json(config, out, outSize, error, errorSize);
    } @catch (NSException *ex) { return exception_error(error, errorSize, ex); } }
}
int metal_run(void *plan, char *error, size_t errorSize) {
    @autoreleasepool { @try {
        Plan *p = static_cast<Plan *>(plan); if (!p) return fail(error, errorSize, @"closed plan");
        if (p->failure) return fail(error, errorSize, p->failure);
        id<MTLCommandBuffer> command = [p->queue commandBuffer];
        if (!command) return fail(error, errorSize, @"could not create command buffer");
        if (encode(p, command, 1, error, errorSize)) return -1;
        [p->pending addObject:command]; [command commit]; return 0;
    } @catch (NSException *ex) { return exception_error(error, errorSize, ex); } }
}
int metal_synchronize(void *plan, char *error, size_t errorSize) {
    @autoreleasepool { @try { Plan *p = static_cast<Plan *>(plan); return p ? sync_plan(p, error, errorSize) : fail(error, errorSize, @"closed plan"); }
    @catch (NSException *ex) { return exception_error(error, errorSize, ex); } }
}
int metal_copy_output(void *plan, float *out, char *error, size_t errorSize) {
    @autoreleasepool { @try {
        Plan *p = static_cast<Plan *>(plan); if (!p || !out) return fail(error, errorSize, @"closed plan or null output");
        if (sync_plan(p, error, errorSize)) return -1;
        if (p->strideC == size_t(p->dims.n) * sizeof(float)) memcpy(out, p->c.contents, size_t(p->dims.m) * p->strideC);
        else for (uint32_t r = 0; r < p->dims.m; ++r) memcpy(out + size_t(r) * p->dims.n, static_cast<char *>(p->c.contents) + size_t(r) * p->strideC, size_t(p->dims.n) * sizeof(float));
        return 0;
    } @catch (NSException *ex) { return exception_error(error, errorSize, ex); } }
}
int metal_samples(void *plan, int warmup, int repeats, int batch, double *out, char *error, size_t errorSize) {
    @autoreleasepool { @try {
        Plan *p = static_cast<Plan *>(plan);
        if (!p || !out || warmup < 0 || repeats <= 0 || batch <= 0) return fail(error, errorSize, @"invalid timing arguments");
        if (sync_plan(p, error, errorSize)) return -1;
        for (int i = -1; i < repeats; ++i) {
            int count = i < 0 ? warmup : batch;
            if (!count) continue;
            id<MTLCommandBuffer> command = [p->queue commandBuffer];
            if (!command) return fail(error, errorSize, @"could not create timing command buffer");
            if (encode(p, command, count, error, errorSize)) return -1;
            [p->pending addObject:command]; [command commit];
            if (sync_plan(p, error, errorSize)) return -1;
            if (i >= 0) {
                double start = command.GPUStartTime, end = command.GPUEndTime;
                double elapsed = (end - start) / batch;
                if (!(start > 0 && end > start && std::isfinite(elapsed) && elapsed > 0)) return fail(error, errorSize, @"Metal GPU timestamps are unavailable or invalid");
                out[i] = elapsed;
            }
        }
        return 0;
    } @catch (NSException *ex) { return exception_error(error, errorSize, ex); } }
}
int metal_plan_destroy(void *plan, char *error, size_t errorSize) {
    @autoreleasepool { @try {
        std::unique_ptr<Plan> p(static_cast<Plan *>(plan));
        return p ? sync_plan(p.get(), error, errorSize) : 0;
    } @catch (NSException *ex) { return exception_error(error, errorSize, ex); } }
}
}
