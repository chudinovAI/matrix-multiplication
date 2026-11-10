#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cstdint>

#include <cstddef>
#include <cstdio>
#include <new>
#include <type_traits>

namespace {

constexpr int TILE = 32;
constexpr int REGISTER_M = 64;
constexpr int REGISTER_N = 64;
constexpr int REGISTER_K = 16;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr unsigned int MAX_GRID = 65535;

struct MatmulContext {
    int device;
    cublasHandle_t handle;
    int capability_major;
    int capability_minor;
    cublasMath_t math_mode = CUBLAS_PEDANTIC_MATH;
    cudaStream_t stream = nullptr;
    // Populated once at context creation, never during graph capture or launch.
    cudaError_t image_status[4] = {};
    int image_ptx_version[4] = {};
};

void clear_error(char *error, size_t error_size)
{
    if (error && error_size)
        error[0] = '\0';
}

int fail(char *error, size_t error_size, const char *operation, const char *detail)
{
    if (error && error_size)
        std::snprintf(error, error_size, "%s: %s", operation, detail);
    return 1;
}

const char *cublas_error(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "library not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "resource allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "unsupported GPU architecture";
    case CUBLAS_STATUS_MAPPING_ERROR: return "memory mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "operation not supported";
    case CUBLAS_STATUS_LICENSE_ERROR: return "license error";
    default: return "unknown cuBLAS error";
    }
}

unsigned int grid_size(int dimension, int tile)
{
    const size_t blocks = (static_cast<size_t>(dimension) + tile - 1) / tile;
    return static_cast<unsigned int>(blocks < MAX_GRID ? blocks : MAX_GRID);
}

// Original shared-memory tiled baseline: one thread computes two columns.
template <int Tile>
__global__ void matmul_tiled_fp32(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C,
                                  int M, int N, int K)
{
    __shared__ float As[Tile][Tile];
    __shared__ float Bs[Tile][Tile * 2];

    for (size_t block_row = blockIdx.y; block_row * Tile < static_cast<size_t>(M);
         block_row += gridDim.y) {
        for (size_t block_col = blockIdx.x; block_col * Tile * 2 < static_cast<size_t>(N);
             block_col += gridDim.x) {
            const size_t row = block_row * Tile + threadIdx.y;
            const size_t col = block_col * Tile * 2 + threadIdx.x * 2;
            float acc0 = 0.0f;
            float acc1 = 0.0f;

            for (size_t k_base = 0; k_base < static_cast<size_t>(K); k_base += Tile) {
                const size_t a_col = k_base + threadIdx.x;
                const size_t b_row = k_base + threadIdx.y;
                As[threadIdx.y][threadIdx.x] =
                    (row < static_cast<size_t>(M) && a_col < static_cast<size_t>(K))
                        ? A[row * K + a_col] : 0.0f;
                Bs[threadIdx.y][threadIdx.x] =
                    (b_row < static_cast<size_t>(K) && col < static_cast<size_t>(N))
                        ? B[b_row * N + col] : 0.0f;
                Bs[threadIdx.y][threadIdx.x + Tile] =
                    (b_row < static_cast<size_t>(K) && col + 1 < static_cast<size_t>(N))
                        ? B[b_row * N + col + 1] : 0.0f;

                __syncthreads();
#pragma unroll
                for (int e = 0; e < Tile; ++e) {
                    const float a_val = As[threadIdx.y][e];
                    acc0 += a_val * Bs[e][threadIdx.x];
                    acc1 += a_val * Bs[e][threadIdx.x + Tile];
                }
                __syncthreads();
            }

            if (row < static_cast<size_t>(M) && col < static_cast<size_t>(N))
                C[row * N + col] = acc0;
            if (row < static_cast<size_t>(M) && col + 1 < static_cast<size_t>(N))
                C[row * N + col + 1] = acc1;
        }
    }
}

// Each 16x16 thread block computes a 64x64 output tile. Threads cooperatively
// load 64x16 and 16x64 tiles, then accumulate a 4x4 output tile in registers.
__global__ void matmul_register_fp32(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C,
                                     int M, int N, int K)
{
    __shared__ float As[REGISTER_M][REGISTER_K];
    __shared__ float Bs[REGISTER_K][REGISTER_N];
    const int tid = threadIdx.y * THREADS_X + threadIdx.x;

    for (size_t block_row = blockIdx.y; block_row * REGISTER_M < static_cast<size_t>(M);
         block_row += gridDim.y) {
        for (size_t block_col = blockIdx.x; block_col * REGISTER_N < static_cast<size_t>(N);
             block_col += gridDim.x) {
            const size_t row_base = block_row * REGISTER_M;
            const size_t col_base = block_col * REGISTER_N;
            float accum[4][4] = {};

            for (size_t k_base = 0; k_base < static_cast<size_t>(K); k_base += REGISTER_K) {
                for (int index = tid; index < REGISTER_M * REGISTER_K;
                     index += THREADS_X * THREADS_Y) {
                    const int local_row = index / REGISTER_K;
                    const int local_k = index % REGISTER_K;
                    const size_t row = row_base + local_row;
                    const size_t k = k_base + local_k;
                    As[local_row][local_k] =
                        (row < static_cast<size_t>(M) && k < static_cast<size_t>(K))
                            ? A[row * K + k] : 0.0f;
                }
                for (int index = tid; index < REGISTER_K * REGISTER_N;
                     index += THREADS_X * THREADS_Y) {
                    const int local_k = index / REGISTER_N;
                    const int local_col = index % REGISTER_N;
                    const size_t k = k_base + local_k;
                    const size_t col = col_base + local_col;
                    Bs[local_k][local_col] =
                        (k < static_cast<size_t>(K) && col < static_cast<size_t>(N))
                            ? B[k * N + col] : 0.0f;
                }
                __syncthreads();

#pragma unroll
                for (int e = 0; e < REGISTER_K; ++e) {
                    float a_values[4];
                    float b_values[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        a_values[i] = As[threadIdx.y + i * THREADS_Y][e];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        b_values[j] = Bs[e][threadIdx.x + j * THREADS_X];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                            accum[i][j] += a_values[i] * b_values[j];
                }
                __syncthreads();
            }

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const size_t row = row_base + threadIdx.y + i * THREADS_Y;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const size_t col = col_base + threadIdx.x + j * THREADS_X;
                    if (row < static_cast<size_t>(M) && col < static_cast<size_t>(N))
                        C[row * N + col] = accum[i][j];
                }
            }
        }
    }
}

#include "matmul_custom.cuh"

template <typename Kernel>
void inspect_image(MatmulContext *context, int index, Kernel kernel)
{
    cudaFuncAttributes attributes{};
    const cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
    context->image_status[index] = status;
    if (status == cudaSuccess)
        context->image_ptx_version[index] = attributes.ptxVersion;
    else
        cudaGetLastError();
}

} // namespace

extern "C" int matmul_abi_version()
{
    return 3;
}

extern "C" int matmul_kernel_info(int implementation, int precision, char *json,
                                   size_t json_size, char *error, size_t error_size)
{
    clear_error(error, error_size);
    if (!json || !json_size)
        return fail(error, error_size, "matmul_kernel_info", "JSON buffer is null or empty");
    json[0] = '\0';
    if (implementation < 0 || implementation > 6 || precision < 0 || precision > 3)
        return fail(error, error_size, "matmul_kernel_info", "unknown implementation or precision");
    if ((implementation != 2 && implementation != 6 && precision != 0) ||
        (implementation == 6 && precision == 0))
        return fail(error, error_size, "matmul_kernel_info", "unsupported implementation and precision");
    const char *names[] = {"tiled", "register", "cublas", "warp", "warp_large", "pipeline", "wmma"};
    const char *precisions[] = {"fp32", "tf32", "fp16", "bf16"};
    int bm = 0, bn = 0, bk = 0, threads = 0, wm = 0, wn = 0, mm = 0, mn = 0;
    int buffers = 0;
    const char *load = "library";
    bool tensor = false;
    switch (implementation) {
    case 0:
        bm = TILE; bn = TILE * 2; bk = TILE; threads = TILE * TILE;
        mm = 1; mn = 2; buffers = 1; load = "scalar"; break;
    case 1:
        bm = REGISTER_M; bn = REGISTER_N; bk = REGISTER_K; threads = THREADS_X * THREADS_Y;
        mm = 4; mn = 4; buffers = 1; load = "scalar"; break;
    case 3:
        bm = WARP_SMALL_M; bn = WARP_N; bk = WARP_K; threads = WARP_THREADS;
        wm = WARP_SMALL_TILE_M; wn = WARP_TILE_N; mm = WARP_SMALL_MICRO_M;
        mn = WARP_MICRO_N; buffers = 1; load = "float4_masked"; break;
    case 4:
        bm = WARP_LARGE_M; bn = WARP_N; bk = WARP_LARGE_K; threads = WARP_THREADS;
        wm = WARP_LARGE_TILE_M; wn = WARP_TILE_N; mm = WARP_LARGE_MICRO_M;
        mn = WARP_MICRO_N; buffers = 1; load = "float4_masked"; break;
    case 5:
        bm = WARP_LARGE_M; bn = WARP_N; bk = WARP_K; threads = WARP_THREADS;
        wm = WARP_LARGE_TILE_M; wn = WARP_TILE_N; mm = WARP_LARGE_MICRO_M;
        mn = WARP_MICRO_N; buffers = 2; load = "cp.async_float4_masked"; break;
    case 6:
        bm = MMA_M; bn = MMA_N; bk = precision == 1 ? MMA_TF32_K : MMA_HALF_K;
        threads = MMA_THREADS; wm = MMA_WARP_M; wn = MMA_WARP_N;
        mm = 16; mn = 16; buffers = 1; load = "shared_masked"; tensor = true; break;
    }
    const int written = std::snprintf(json, json_size,
        "{\"implementation\":\"%s\",\"precision\":\"%s\",\"tile_m\":%d,"
        "\"tile_n\":%d,\"tile_k\":%d,\"threads\":%d,\"warp_tile_m\":%d,"
        "\"warp_tile_n\":%d,\"microtile_m\":%d,\"microtile_n\":%d,"
        "\"buffering\":%d,\"load_method\":\"%s\",\"tensor_cores\":%s}",
        names[implementation], precisions[precision], bm, bn, bk, threads, wm, wn,
        mm, mn, buffers, load, implementation == 2 ? "null" : tensor ? "true" : "false");
    if (written < 0 || static_cast<size_t>(written) >= json_size) {
        json[0] = '\0';
        return fail(error, error_size, "matmul_kernel_info", "JSON buffer too small");
    }
    return 0;
}

extern "C" void *matmul_create(int device, char *error, size_t error_size)
{
    clear_error(error, error_size);
    cudaError_t cuda_status = cudaSetDevice(device);
    if (cuda_status != cudaSuccess) {
        fail(error, error_size, "cudaSetDevice", cudaGetErrorString(cuda_status));
        // The failure is returned above; consume its thread-local CUDA error
        // so a later valid context does not report it as a new kernel failure.
        cudaGetLastError();
        return nullptr;
    }

    int major = 0;
    int minor = 0;
    cuda_status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (cuda_status == cudaSuccess)
        cuda_status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (cuda_status != cudaSuccess) {
        fail(error, error_size, "cudaDeviceGetAttribute", cudaGetErrorString(cuda_status));
        cudaGetLastError();
        return nullptr;
    }

    MatmulContext *context = new (std::nothrow) MatmulContext{device, nullptr, major, minor};
    if (!context) {
        fail(error, error_size, "matmul_create", "could not allocate context");
        return nullptr;
    }
    cublasStatus_t status = cublasCreate(&context->handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail(error, error_size, "cublasCreate", cublas_error(status));
        delete context;
        return nullptr;
    }
    status = cublasSetMathMode(context->handle, CUBLAS_PEDANTIC_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
        const cublasStatus_t destroy_status = cublasDestroy(context->handle);
        if (destroy_status != CUBLAS_STATUS_SUCCESS && error && error_size)
            std::snprintf(error, error_size, "cublasSetMathMode: %s; cublasDestroy: %s",
                          cublas_error(status), cublas_error(destroy_status));
        else
            fail(error, error_size, "cublasSetMathMode", cublas_error(status));
        delete context;
        return nullptr;
    }
    if (major >= 8) {
        inspect_image(context, 0,
            matmul_warp_fp32<WARP_LARGE_M, WARP_K, WARP_LARGE_MICRO_M, true>);
        inspect_image(context, 1,
            matmul_wmma<float, MMA_TF32_K>);
        inspect_image(context, 2, matmul_wmma<__half, MMA_HALF_K>);
        inspect_image(context, 3, matmul_wmma<__nv_bfloat16, MMA_HALF_K>);
    }
    return context;
}

extern "C" int matmul_destroy(void *opaque, char *error, size_t error_size)
{
    clear_error(error, error_size);
    if (!opaque)
        return fail(error, error_size, "matmul_destroy", "context is null");
    MatmulContext *context = static_cast<MatmulContext *>(opaque);
    const cudaError_t cuda_status = cudaSetDevice(context->device);
    if (cuda_status != cudaSuccess) {
        fail(error, error_size, "cudaSetDevice", cudaGetErrorString(cuda_status));
        cudaGetLastError();
        return 1;
    }
    const cublasStatus_t status = cublasDestroy(context->handle);
    if (status != CUBLAS_STATUS_SUCCESS)
        return fail(error, error_size, "cublasDestroy", cublas_error(status));
    delete context;
    return 0;
}

extern "C" int matmul_launch(void *opaque, int implementation, int precision,
                             const void *a, const void *b, float *c,
                             int m, int k, int n, void *stream,
                             char *error, size_t error_size)
{
    clear_error(error, error_size);
    if (!opaque)
        return fail(error, error_size, "matmul_launch", "context is null");
    if (!a || !b || !c)
        return fail(error, error_size, "matmul_launch", "matrix pointer is null");
    if (m <= 0 || k <= 0 || n <= 0)
        return fail(error, error_size, "matmul_launch", "dimensions must be positive");
    if (implementation < 0 || implementation > 6)
        return fail(error, error_size, "matmul_launch", "unknown implementation");
    if (precision < 0 || precision > 3)
        return fail(error, error_size, "matmul_launch", "unknown precision");
    if ((implementation != 2 && implementation != 6 && precision != 0) ||
        (implementation == 6 && precision == 0))
        return fail(error, error_size, "matmul_launch", "unsupported implementation and precision");

    MatmulContext *context = static_cast<MatmulContext *>(opaque);
    if ((implementation == 5 || implementation == 6) && context->capability_major < 8)
        return fail(error, error_size, "matmul_launch", "pipeline and wmma require compute capability 8.0");
    if ((precision == 1 || precision == 3) && context->capability_major < 8)
        return fail(error, error_size, "matmul_launch", "TF32 and BF16 require compute capability 8.0");
    if (precision == 2 &&
        (context->capability_major < 5 ||
         (context->capability_major == 5 && context->capability_minor < 3)))
        return fail(error, error_size, "matmul_launch", "FP16 requires compute capability 5.3");
    if (implementation == 5 || implementation == 6) {
        const int index = implementation == 5 ? 0 : precision;
        if (context->image_status[index] != cudaSuccess) {
            if (error && error_size)
                std::snprintf(error, error_size,
                              "matmul_launch: cannot inspect CUDA kernel image: %s; "
                              "rebuild with compute capability 8.0 or newer",
                              cudaGetErrorString(context->image_status[index]));
            return 1;
        }
        if (context->image_ptx_version[index] < 80)
            return fail(error, error_size, "matmul_launch",
                        "kernel image was compiled for a pre-sm80 virtual architecture; "
                        "rebuild with -arch=sm_80 or newer");
    }
    const cudaError_t device_status = cudaSetDevice(context->device);
    if (device_status != cudaSuccess) {
        fail(error, error_size, "cudaSetDevice", cudaGetErrorString(device_status));
        cudaGetLastError();
        return 1;
    }
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    if (implementation == 2) {
        cublasStatus_t status;
        const cublasMath_t math_mode = precision == 0
            ? CUBLAS_PEDANTIC_MATH
            : static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH |
                                       CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        if (context->math_mode != math_mode) {
            status = cublasSetMathMode(context->handle, math_mode);
            if (status != CUBLAS_STATUS_SUCCESS)
                return fail(error, error_size, "cublasSetMathMode", cublas_error(status));
            context->math_mode = math_mode;
        }
        if (context->stream != cuda_stream) {
            status = cublasSetStream(context->handle, cuda_stream);
            if (status != CUBLAS_STATUS_SUCCESS)
                return fail(error, error_size, "cublasSetStream", cublas_error(status));
            context->stream = cuda_stream;
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // C^T = B^T A^T: reinterpret row-major buffers as column-major.
        if (precision == 0) {
            status = cublasSgemm(context->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, m, k, &alpha, static_cast<const float *>(b), n,
                                 static_cast<const float *>(a), k, &beta, c, n);
        } else {
            const cudaDataType_t input_type = precision == 2 ? CUDA_R_16F
                                            : precision == 3 ? CUDA_R_16BF : CUDA_R_32F;
            const cublasComputeType_t compute_type = precision == 1
                ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
            status = cublasGemmEx(context->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, m, k, &alpha, b, input_type, n, a, input_type, k,
                                  &beta, c, CUDA_R_32F, n, compute_type, CUBLAS_GEMM_DEFAULT);
        }
        if (status != CUBLAS_STATUS_SUCCESS)
            return fail(error, error_size, precision == 0 ? "cublasSgemm" : "cublasGemmEx",
                        cublas_error(status));
        return 0;
    }

    if (implementation == 0) {
        const dim3 block(TILE, TILE);
        const dim3 grid(grid_size(n, TILE * 2), grid_size(m, TILE));
        matmul_tiled_fp32<TILE><<<grid, block, 0, cuda_stream>>>(
            static_cast<const float *>(a), static_cast<const float *>(b), c, m, n, k);
    } else if (implementation == 1) {
        const dim3 block(THREADS_X, THREADS_Y);
        const dim3 grid(grid_size(n, REGISTER_N), grid_size(m, REGISTER_M));
        matmul_register_fp32<<<grid, block, 0, cuda_stream>>>(
            static_cast<const float *>(a), static_cast<const float *>(b), c, m, n, k);
    } else if (implementation >= 3 && implementation <= 5) {
        const int bm = implementation == 3 ? WARP_SMALL_M : WARP_LARGE_M;
        const dim3 grid(grid_size(n, WARP_N), grid_size(m, bm));
        if (implementation == 3)
            matmul_warp_fp32<WARP_SMALL_M, WARP_K, WARP_SMALL_MICRO_M, false>
                <<<grid, WARP_THREADS, 0, cuda_stream>>>(static_cast<const float *>(a), static_cast<const float *>(b), c, m, n, k);
        else if (implementation == 4)
            matmul_warp_fp32<WARP_LARGE_M, WARP_LARGE_K, WARP_LARGE_MICRO_M, false>
                <<<grid, WARP_THREADS, 0, cuda_stream>>>(static_cast<const float *>(a), static_cast<const float *>(b), c, m, n, k);
        else
            matmul_warp_fp32<WARP_LARGE_M, WARP_K, WARP_LARGE_MICRO_M, true>
                <<<grid, WARP_THREADS, 0, cuda_stream>>>(static_cast<const float *>(a), static_cast<const float *>(b), c, m, n, k);
    } else {
        const dim3 grid(grid_size(n, MMA_N), grid_size(m, MMA_M));
        if (precision == 1)
            matmul_wmma<float, MMA_TF32_K>
                <<<grid, MMA_THREADS, 0, cuda_stream>>>(static_cast<const float *>(a), static_cast<const float *>(b), c, m, n, k);
        else if (precision == 2)
            matmul_wmma<__half, MMA_HALF_K>
                <<<grid, MMA_THREADS, 0, cuda_stream>>>(static_cast<const __half *>(a), static_cast<const __half *>(b), c, m, n, k);
        else
            matmul_wmma<__nv_bfloat16, MMA_HALF_K>
                <<<grid, MMA_THREADS, 0, cuda_stream>>>(static_cast<const __nv_bfloat16 *>(a), static_cast<const __nv_bfloat16 *>(b), c, m, n, k);
    }
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess)
        return fail(error, error_size, "kernel launch", cudaGetErrorString(launch_status));
    return 0;
}
