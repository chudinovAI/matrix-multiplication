// Included inside matmul.cu's anonymous namespace. All geometry below is also
// used by matmul_kernel_info, so the reported configuration follows the code.
constexpr int WARP_THREADS = 256;
constexpr int WARP_N = 128;
constexpr int WARP_K = 16;
constexpr int WARP_SMALL_M = 64;
constexpr int WARP_LARGE_M = 128;
constexpr int WARP_LARGE_K = 32;
constexpr int WARP_TILE_N = 32;
constexpr int WARP_SMALL_TILE_M = 32;
constexpr int WARP_LARGE_TILE_M = 64;
constexpr int WARP_MICRO_N = 8;
constexpr int WARP_SMALL_MICRO_M = 4;
constexpr int WARP_LARGE_MICRO_M = 8;
constexpr int MMA_M = 128;
constexpr int MMA_N = 128;
constexpr int MMA_THREADS = 256;
constexpr int MMA_WARP_M = 32;
constexpr int MMA_WARP_N = 64;
constexpr int MMA_HALF_K = 16;
constexpr int MMA_TF32_K = 8;

__device__ __forceinline__ bool aligned16(const void *p)
{
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// Each iteration moves four adjacent values; the scalar path handles both
// unaligned row starts and partial vectors without reading out of bounds.
template <int BM, int BK, bool Async>
__device__ __forceinline__ void stage_simt(float (*as)[BK + 4], float (*bs)[WARP_N],
                                           const float *a, const float *b,
                                           size_t rb, size_t cb, size_t kb,
                                           int m, int n, int k)
{
    for (int v = threadIdx.x; v < BM * BK / 4; v += WARP_THREADS) {
        int row = v / (BK / 4), x = (v % (BK / 4)) * 4;
        size_t gr = rb + row, gk = kb + x;
        float *dst = &as[row][x];
        if (gr < static_cast<size_t>(m) && gk + 3 < static_cast<size_t>(k)) {
            const float *src = a + gr * k + gk;
            if (aligned16(src)) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                if (Async) {
                    unsigned int smem = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem), "l"(src) : "memory");
                } else
#endif
                    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(src);
                continue;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            dst[j] = gr < static_cast<size_t>(m) && gk + j < static_cast<size_t>(k)
                ? a[gr * k + gk + j] : 0.0f;
    }
    for (int v = threadIdx.x; v < BK * WARP_N / 4; v += WARP_THREADS) {
        int row = v / (WARP_N / 4), x = (v % (WARP_N / 4)) * 4;
        size_t gk = kb + row, gc = cb + x;
        float *dst = &bs[row][x];
        if (gk < static_cast<size_t>(k) && gc + 3 < static_cast<size_t>(n)) {
            const float *src = b + gk * n + gc;
            if (aligned16(src)) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                if (Async) {
                    unsigned int smem = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem), "l"(src) : "memory");
                } else
#endif
                    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(src);
                continue;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            dst[j] = gk < static_cast<size_t>(k) && gc + j < static_cast<size_t>(n)
                ? b[gk * n + gc + j] : 0.0f;
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (Async) asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <int BM, int BK, int MICRO_M, bool Async>
__global__ void matmul_warp_fp32(const float *__restrict__ a, const float *__restrict__ b,
                                 float *__restrict__ c, int m, int n, int k)
{
    __shared__ __align__(16) float as[Async ? 2 : 1][BM][BK + 4];
    __shared__ __align__(16) float bs[Async ? 2 : 1][BK][WARP_N];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    if (Async) { asm volatile("trap;"); return; }
#endif
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row0 = (warp / 4) * (BM / 2) + lane / 4;
    const int col0 = (warp % 4) * WARP_TILE_N + (lane % 4) * 4;
    for (size_t br = blockIdx.y; br * BM < static_cast<size_t>(m); br += gridDim.y) {
        for (size_t bc = blockIdx.x; bc * WARP_N < static_cast<size_t>(n); bc += gridDim.x) {
            const size_t rb = br * BM, cb = bc * WARP_N;
            float acc[MICRO_M][WARP_MICRO_N] = {};
            if (Async) stage_simt<BM, BK, true>(as[0], bs[0], a, b, rb, cb, 0, m, n, k);
            int buffer = 0;
            for (size_t kb = 0; kb < static_cast<size_t>(k); kb += BK) {
                if (Async) {
                    if (kb + BK < static_cast<size_t>(k)) {
                        stage_simt<BM, BK, true>(as[buffer ^ 1], bs[buffer ^ 1], a, b,
                                                 rb, cb, kb + BK, m, n, k);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                        asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
                    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                        asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
                    }
                } else {
                    stage_simt<BM, BK, false>(as[0], bs[0], a, b, rb, cb, kb, m, n, k);
                }
                __syncthreads();
#pragma unroll
                for (int x = 0; x < BK; ++x) {
                    float av[MICRO_M], bv[WARP_MICRO_N];
#pragma unroll
                    for (int i = 0; i < MICRO_M; ++i) av[i] = as[buffer][row0 + i * 8][x];
#pragma unroll
                    for (int j = 0; j < WARP_MICRO_N; ++j)
                        bv[j] = bs[buffer][x][col0 + (j / 4) * 16 + j % 4];
#pragma unroll
                    for (int i = 0; i < MICRO_M; ++i)
#pragma unroll
                        for (int j = 0; j < WARP_MICRO_N; ++j) acc[i][j] += av[i] * bv[j];
                }
                __syncthreads();
                if (Async) buffer ^= 1;
            }
#pragma unroll
            for (int i = 0; i < MICRO_M; ++i) {
                size_t r = rb + row0 + i * 8;
#pragma unroll
                for (int j = 0; j < WARP_MICRO_N; j += 4) {
                    size_t col = cb + col0 + (j / 4) * 16;
                    if (r < static_cast<size_t>(m) && col + 3 < static_cast<size_t>(n) &&
                        aligned16(c + r * n + col)) {
                        *reinterpret_cast<float4 *>(c + r * n + col) =
                            make_float4(acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]);
                    } else {
#pragma unroll
                        for (int t = 0; t < 4; ++t)
                            if (r < static_cast<size_t>(m) && col + t < static_cast<size_t>(n))
                                c[r * n + col + t] = acc[i][j+t];
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ float round_tf32_even(float x)
{
    unsigned int bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0xfffu + ((bits >> 13) & 1u)) & 0xffffe000u;
    return __uint_as_float(bits);
}

template <typename T> __device__ __forceinline__ T mma_input(T x) { return x; }
template <> __device__ __forceinline__ float mma_input(float x) { return round_tf32_even(x); }

template <typename T, int BK>
__global__ void matmul_wmma(const T *__restrict__ a, const T *__restrict__ b,
                            float *__restrict__ c, int m, int n, int k)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    using FragmentType = typename std::conditional<std::is_same<T, float>::value,
                                                   wmma::precision::tf32, T>::type;
    __shared__ __align__(32) T as[MMA_M][BK];
    __shared__ __align__(32) T bs[BK][MMA_N];
    __shared__ __align__(32) float scratch[8][16 * 16];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wr = (warp / 2) * MMA_WARP_M, wc = (warp % 2) * MMA_WARP_N;
    for (size_t br = blockIdx.y; br * MMA_M < static_cast<size_t>(m); br += gridDim.y) {
        for (size_t bc = blockIdx.x; bc * MMA_N < static_cast<size_t>(n); bc += gridDim.x) {
            size_t rb = br * MMA_M, cb = bc * MMA_N;
            wmma::fragment<wmma::accumulator, 16, 16, BK, float> acc[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
            for (size_t kb = 0; kb < static_cast<size_t>(k); kb += BK) {
                for (int x = threadIdx.x; x < MMA_M * BK; x += MMA_THREADS) {
                    int r = x / BK, q = x % BK;
                    as[r][q] = rb + r < static_cast<size_t>(m) && kb + q < static_cast<size_t>(k)
                        ? mma_input(a[(rb + r) * k + kb + q]) : T{};
                }
                for (int x = threadIdx.x; x < BK * MMA_N; x += MMA_THREADS) {
                    int q = x / MMA_N, col = x % MMA_N;
                    bs[q][col] = kb + q < static_cast<size_t>(k) && cb + col < static_cast<size_t>(n)
                        ? mma_input(b[(kb + q) * n + cb + col]) : T{};
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16, 16, BK, FragmentType, wmma::row_major> af[2];
                wmma::fragment<wmma::matrix_b, 16, 16, BK, FragmentType, wmma::row_major> bf[4];
#pragma unroll
                for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(af[i], &as[wr + i * 16][0], BK);
#pragma unroll
                for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(bf[j], &bs[0][wc + j * 16], MMA_N);
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
                __syncthreads();
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::store_matrix_sync(scratch[warp], acc[i][j], 16, wmma::mem_row_major);
                    __syncwarp();
                    for (int x = lane; x < 256; x += 32) {
                        size_t r = rb + wr + i * 16 + x / 16;
                        size_t col = cb + wc + j * 16 + x % 16;
                        if (r < static_cast<size_t>(m) && col < static_cast<size_t>(n))
                            c[r * n + col] = scratch[warp][x];
                    }
                    __syncwarp();
                }
            }
            __syncthreads();
        }
    }
#else
    asm volatile("trap;");
#endif
}
