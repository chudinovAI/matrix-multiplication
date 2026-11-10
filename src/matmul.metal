#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

// Geometry macros come from the same native constants as dispatch and metadata.
struct Dimensions { uint m, n, k; };

template<typename T>
inline void naive_impl(device const T *a, device const T *b, device float *c,
                       constant Dimensions &d, uint2 p) {
    if (p.x >= d.n || p.y >= d.m) return;
    float sum = 0.0f;
    for (uint k = 0; k < d.k; ++k)
        sum = fma(float(a[ulong(p.y) * d.k + k]), float(b[ulong(k) * d.n + p.x]), sum);
    c[ulong(p.y) * d.n + p.x] = sum;
}
#define NAIVE(NAME, TYPE) \
kernel void NAME(device const TYPE *a [[buffer(0)]], device const TYPE *b [[buffer(1)]], \
                 device float *c [[buffer(2)]], constant Dimensions &d [[buffer(3)]], \
                 uint2 p [[thread_position_in_grid]]) { naive_impl(a, b, c, d, p); }
NAIVE(naive_f32, float)
NAIVE(naive_f16, half)

template<typename T>
inline void tiled_impl(device const T *a, device const T *b, device float *c,
                       constant Dimensions &d, uint2 group, uint2 local,
                       threadgroup float *as, threadgroup float *bs) {
    const uint tid = local.y * TILED_X + local.x;
    const uint row0 = group.y * TILED_M, col0 = group.x * TILED_N;
    float sums[TILED_M / TILED_Y][TILED_N / TILED_X] = {};
    for (uint base = 0; base < d.k; base += TILED_K) {
        for (uint i = tid; i < TILED_M * TILED_K; i += TILED_X * TILED_Y) {
            uint row = i / TILED_K, k = i % TILED_K;
            as[i] = row0 + row < d.m && base + k < d.k ? float(a[ulong(row0 + row) * d.k + base + k]) : 0.0f;
        }
        for (uint i = tid; i < TILED_K * TILED_N; i += TILED_X * TILED_Y) {
            uint k = i / TILED_N, col = i % TILED_N;
            bs[i] = base + k < d.k && col0 + col < d.n ? float(b[ulong(base + k) * d.n + col0 + col]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILED_K; ++k)
            for (uint i = 0; i < TILED_M / TILED_Y; ++i)
                for (uint j = 0; j < TILED_N / TILED_X; ++j)
                    sums[i][j] = fma(as[(local.y + i * TILED_Y) * TILED_K + k],
                                     bs[k * TILED_N + local.x + j * TILED_X], sums[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < TILED_M / TILED_Y; ++i) for (uint j = 0; j < TILED_N / TILED_X; ++j) {
        uint row = row0 + local.y + i * TILED_Y, col = col0 + local.x + j * TILED_X;
        if (row < d.m && col < d.n) c[ulong(row) * d.n + col] = sums[i][j];
    }
}
#define TILED(NAME, TYPE) \
kernel void NAME(device const TYPE *a [[buffer(0)]], device const TYPE *b [[buffer(1)]], \
                 device float *c [[buffer(2)]], constant Dimensions &d [[buffer(3)]], \
                 uint2 group [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]]) { \
    threadgroup float as[TILED_M * TILED_K], bs[TILED_K * TILED_N]; \
    tiled_impl(a, b, c, d, group, local, as, bs); }
TILED(tiled_f32, float)
TILED(tiled_f16, half)

template<typename T>
inline void simd_impl(device const T *a, device const T *b, device float *c,
                      constant Dimensions &d, uint2 group, uint tid, uint sg,
                      threadgroup T *as, threadgroup T *bs, threadgroup float *scratch) {
    const uint row0 = group.y * SIMD_M, col0 = group.x * SIMD_N;
    const uint sr = (sg / (SIMD_N / SIMD_WN)) * SIMD_WM;
    const uint sc = (sg % (SIMD_N / SIMD_WN)) * SIMD_WN;
    simdgroup_float8x8 acc[SIMD_WM / 8][SIMD_WN / 8];
    for (uint i = 0; i < SIMD_WM / 8; ++i) for (uint j = 0; j < SIMD_WN / 8; ++j)
        acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint base = 0; base < d.k; base += SIMD_K) {
        for (uint i = tid; i < SIMD_M * SIMD_K; i += SIMD_THREADS) {
            uint row = i / SIMD_K, k = i % SIMD_K;
            as[i] = row0 + row < d.m && base + k < d.k ? a[ulong(row0 + row) * d.k + base + k] : T(0);
        }
        for (uint i = tid; i < SIMD_K * SIMD_N; i += SIMD_THREADS) {
            uint k = i / SIMD_N, col = i % SIMD_N;
            bs[i] = base + k < d.k && col0 + col < d.n ? b[ulong(base + k) * d.n + col0 + col] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < SIMD_K; k += 8) {
            simdgroup_matrix<T, 8, 8> av[SIMD_WM / 8], bv[SIMD_WN / 8];
            for (uint i = 0; i < SIMD_WM / 8; ++i) simdgroup_load(av[i], as + (sr + i * 8) * SIMD_K + k, SIMD_K);
            for (uint j = 0; j < SIMD_WN / 8; ++j) simdgroup_load(bv[j], bs + k * SIMD_N + sc + j * 8, SIMD_N);
            for (uint i = 0; i < SIMD_WM / 8; ++i) for (uint j = 0; j < SIMD_WN / 8; ++j)
                simdgroup_multiply_accumulate(acc[i][j], av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const bool full = row0 + SIMD_M <= d.m && col0 + SIMD_N <= d.n;
    for (uint i = 0; i < SIMD_WM / 8; ++i) for (uint j = 0; j < SIMD_WN / 8; ++j) {
        uint row = row0 + sr + i * 8, col = col0 + sc + j * 8;
        if (full) simdgroup_store(acc[i][j], c + ulong(row) * d.n + col, d.n);
        else {
            // Only one8x8 fragment per SIMD group, not an entire64x64 output tile.
            threadgroup float *tile = scratch + sg * 64;
            simdgroup_store(acc[i][j], tile, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint x = tid % 32; x < 64; x += 32)
                if (row + x / 8 < d.m && col + x % 8 < d.n)
                    c[ulong(row + x / 8) * d.n + col + x % 8] = tile[x];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
#define SIMD(NAME, TYPE) \
kernel void NAME(device const TYPE *a [[buffer(0)]], device const TYPE *b [[buffer(1)]], \
                 device float *c [[buffer(2)]], constant Dimensions &d [[buffer(3)]], \
                 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup TYPE as[SIMD_M * SIMD_K], bs[SIMD_K * SIMD_N]; \
    threadgroup float scratch[(SIMD_THREADS / 32) * 64]; \
    simd_impl(a, b, c, d, group, tid, sg, as, bs, scratch); }
SIMD(simd_f32, float)
SIMD(simd_f16, half)

#if __METAL_VERSION__ >= 400
using namespace mpp::tensor_ops;

template<typename A, typename B, typename C>
inline void tensor_product(A a, B b, C c) {
    // relaxed_precision=false; the cooperative destination is always FP32.
    constexpr auto descriptor = matmul2d_descriptor(TENSOR_M, TENSOR_N, dynamic_length_v<int>, false, false, false);
    matmul2d<descriptor, execution_simdgroups<TENSOR_SIMDGROUPS>> operation;
    auto result = operation.get_destination_cooperative_tensor<A, B, float>();
    operation.run(a, b, result);
    result.store(c);
}

template<typename T>
inline void tensor_impl(device T *a, device T *b, device float *c, constant Dimensions &d, uint2 tile) {
    const int row = int(tile.y * TENSOR_M), col = int(tile.x * TENSOR_N);
    auto A = tensor(a, dextents<int, 2>{int(d.k), int(d.m)}, array<int, 2>{1, int(d.k)});
    auto B = tensor(b, dextents<int, 2>{int(d.n), int(d.k)}, array<int, 2>{1, int(d.n)});
    auto C = tensor(c, dextents<int, 2>{int(d.n), int(d.m)}, array<int, 2>{1, int(d.n)});
    if (ulong(row) + TENSOR_M <= d.m && ulong(col) + TENSOR_N <= d.n)
        tensor_product(A.template slice<dynamic_extent, TENSOR_M>(0, row),
                       B.template slice<TENSOR_N, dynamic_extent>(col, 0), C.slice<TENSOR_N, TENSOR_M>(col, row));
    else tensor_product(A.slice(0, row), B.slice(col, 0), C.slice(col, row));
}

inline uint2 morton_tile(uint index, constant Dimensions &d) {
    const uint nx = (d.n + TENSOR_N - 1) / TENSOR_N, ny = (d.m + TENSOR_M - 1) / TENSOR_M;
    if (nx < MORTON_SIDE || ny < MORTON_SIDE) return uint2(index % nx, index / nx);
    const uint macro = index / (MORTON_SIDE * MORTON_SIDE), local = index % (MORTON_SIDE * MORTON_SIDE);
    uint2 tile = uint2(macro % ((nx + MORTON_SIDE - 1) / MORTON_SIDE), macro / ((nx + MORTON_SIDE - 1) / MORTON_SIDE)) * MORTON_SIDE;
    for (uint bit = 1, shift = 0; bit < MORTON_SIDE; bit <<= 1, shift += 2)
        tile += uint2((local >> shift) & 1, (local >> (shift + 1)) & 1) * bit;
    return tile;
}
#define TENSOR(NAME, TYPE, MORTON) \
kernel void NAME(device TYPE *a [[buffer(0)]], device TYPE *b [[buffer(1)]], \
                 device float *c [[buffer(2)]], constant Dimensions &d [[buffer(3)]], \
                 uint2 group [[threadgroup_position_in_grid]]) { \
    uint2 tile = MORTON ? morton_tile(group.x, d) : group; \
    if (ulong(tile.x) * TENSOR_N >= d.n || ulong(tile.y) * TENSOR_M >= d.m) return; \
    tensor_impl(a, b, c, d, tile); }
TENSOR(tensor_f32, float, false)
TENSOR(tensor_f16, half, false)
TENSOR(tensor_bf16, bfloat, false)
TENSOR(tensor_morton_f32, float, true)
TENSOR(tensor_morton_f16, half, true)
TENSOR(tensor_morton_bf16, bfloat, true)
#endif
