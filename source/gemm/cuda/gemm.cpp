/**
 * General Matrix Multiplication (GEMM) - CUDA Implementation
 *
 * Computes: C = α * A * B + β * C
 * Where A is MxK, B is KxN, C is MxN
 *
 * GPU Patterns Used:
 * - Tiled Partitioning: Loading tiles into shared memory for data reuse
 * - Map: Each thread computes one (or more) output elements
 * - Reduction: Accumulating partial products along K dimension
 *
 * Precision: FP32
 */

#include "gemm.h"
#include "cuda_helpers.h"
#include <cmath>

/**
 * Kernel 1: Basic Tiled GEMM
 *
 * Each thread block computes a TILE_SIZE x TILE_SIZE tile of C.
 * Uses shared memory to cache tiles of A and B for reuse.
 *
 * Pattern: Tiled Partitioning + Map + Reduction
 */
__global__ void gemmTiled(const float* A, const float* B, float* C,
                           float alpha, float beta,
                           int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread position within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column for this thread's output element
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator (FP32 for precision)
    float sum = 0.0f;

    // Loop over tiles along K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result: C = α * (A * B) + β * C
    if (row < M && col < N) {
        float c_val = C[row * N + col];
        float result = alpha * sum + beta * c_val;
        C[row * N + col] = result;
    }
}

/**
 * Kernel 2: Tiled GEMM with 2x2 thread tiling (better arithmetic intensity)
 *
 * Each thread computes a 2x2 block of output elements.
 * Increases arithmetic intensity and register usage.
 */
__global__ void gemmTiled2x2(const float* A, const float* B, float* C,
                              float alpha, float beta,
                              int M, int N, int K) {
    __shared__ float As[TILE_K][TILE_M];
    __shared__ float Bs[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each thread computes 2x2 output elements
    int rowBase = blockIdx.y * TILE_M + ty * THREAD_TILE;
    int colBase = blockIdx.x * TILE_N + tx * THREAD_TILE;

    // Accumulators for 2x2 output tile
    float sum[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    int numTiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < numTiles; t++) {
        // Cooperative loading of A tile
        // Each thread loads multiple elements
        for (int i = 0; i < THREAD_TILE; i++) {
            int row = rowBase + i;
            int col = t * TILE_K + tx;
            if (row < M && col < K) {
                As[tx][ty * THREAD_TILE + i] = A[row * K + col];
            } else {
                As[tx][ty * THREAD_TILE + i] = 0.0f;
            }
        }

        // Cooperative loading of B tile
        for (int i = 0; i < THREAD_TILE; i++) {
            int row = t * TILE_K + ty;
            int col = colBase + i;
            if (row < K && col < N) {
                Bs[ty][tx * THREAD_TILE + i] = B[row * N + col];
            } else {
                Bs[ty][tx * THREAD_TILE + i] = 0.0f;
            }
        }

        __syncthreads();

        // Compute 2x2 output tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a0 = As[k][ty * THREAD_TILE + 0];
            float a1 = As[k][ty * THREAD_TILE + 1];
            float b0 = Bs[k][tx * THREAD_TILE + 0];
            float b1 = Bs[k][tx * THREAD_TILE + 1];

            sum[0][0] += a0 * b0;
            sum[0][1] += a0 * b1;
            sum[1][0] += a1 * b0;
            sum[1][1] += a1 * b1;
        }

        __syncthreads();
    }

    // Write 2x2 results
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = rowBase + i;
            int col = colBase + j;
            if (row < M && col < N) {
                float c_val = C[row * N + col];
                float result = alpha * sum[i][j] + beta * c_val;
                C[row * N + col] = result;
            }
        }
    }
}

/**
 * Kernel 3: Optimized with vectorized loads and better memory coalescing
 */
__global__ void gemmOptimized(const float* A, const float* B, float* C,
                               float alpha, float beta,
                               int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Precompute base pointers
    const float* A_row = A + row * K;
    const float* B_col = B + col;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tileOffset = t * TILE_SIZE;

        // Coalesced load of A (row-major, consecutive threads read consecutive elements)
        int aIdx = tileOffset + tx;
        As[ty][tx] = (row < M && aIdx < K) ? A_row[aIdx] : 0.0f;

        // Coalesced load of B
        int bIdx = (tileOffset + ty) * N;
        Bs[ty][tx] = (tileOffset + ty < K && col < N) ? B_col[bIdx] : 0.0f;

        __syncthreads();

        // Compute with loop unrolling
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fmaf(As[ty][k], Bs[k][tx], sum);  // Fused multiply-add
        }

        __syncthreads();
    }

    // Write result with FMA
    if (row < M && col < N) {
        float c_val = C[row * N + col];
        C[row * N + col] = fmaf(alpha, sum, beta * c_val);
    }
}

void gemmGPU(const float* h_A, const float* h_B, float* h_C,
             float alpha, float beta, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemmTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// CPU matrix multiplication reference
void gemmCpu(const float* A, const float* B, float* C,
             float alpha, float beta,
             int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            float c_val = C[i * N + j];
            C[i * N + j] = alpha * sum + beta * c_val;
        }
    }
}
