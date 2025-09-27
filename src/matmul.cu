#include <cuda_runtime.h>
#include <stdio.h>

constexpr int TILE = 32;

template <int TILE>
__global__ void matmul_tiled_fp32(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C,
                                  int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE * 2];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE * 2 + threadIdx.x * 2;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && (col + 0) < N) ? B[b_row * N + (col + 0)] : 0.0f;
        Bs[threadIdx.y][threadIdx.x + TILE] =
            (b_row < K && (col + 1) < N) ? B[b_row * N + (col + 1)] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int e = 0; e < TILE; ++e)
        {
            float a_val = As[threadIdx.y][e];
            acc0 += a_val * Bs[e][threadIdx.x];
            acc1 += a_val * Bs[e][threadIdx.x + TILE];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc0;
    if (row < M && (col + 1) < N)
        C[row * N + (col + 1)] = acc1;
}

extern "C" void matmul(const float *hA, const float *hB, float *hC,
                       int M, int K, int N)
{
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE * 2 - 1) / (TILE * 2), (M + TILE - 1) / TILE);
    matmul_tiled_fp32<TILE><<<grid, block>>>(dA, dB, dC, M, N, K);

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
