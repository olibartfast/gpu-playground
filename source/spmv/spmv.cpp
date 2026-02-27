/**
 * Sparse Matrix-Vector Multiplication (SpMV) - CUDA Implementation
 *
 * Computes: y = A * x where A is a sparse matrix stored in dense row-major format
 *
 * GPU Patterns Used:
 * - Gather: Reading sparse matrix elements from arbitrary positions
 * - Reduction: Summing products within each row
 * - Map: Independent computation per row
 */

#include "spmv.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

/**
 * Kernel 1: One thread per row (simple, good for short rows)
 */
__global__ void spmvRowPerThread(const float* __restrict__ A,
                                  const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        int rowStart = row * N;

        for (int col = 0; col < N; col++) {
            float val = A[rowStart + col];
            if (val != 0.0f) {
                sum += val * x[col];
            }
        }

        y[row] = sum;
    }
}

/**
 * Kernel 2: Warp per row (better for longer rows)
 * Uses __shfl_down_sync with explicit full-warp mask.
 */
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
         val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

__global__ void spmvWarpPerRow(const float* __restrict__ A,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                int M, int N) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warpId < M) {
        float sum = 0.0f;
        int rowStart = warpId * N;

        for (int col = lane; col < N; col += WARP_SIZE) {
            float val = A[rowStart + col];
            if (val != 0.0f) {
                sum += val * x[col];
            }
        }

        sum = warpReduceSum(sum);

        if (lane == 0) {
            y[warpId] = sum;
        }
    }
}

/**
 * Kernel 3: Block per row with shared memory reduction
 */
__global__ void spmvBlockPerRow(const float* __restrict__ A,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int M, int N) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (row < M) {
        float sum = 0.0f;
        int rowStart = row * N;

        for (int col = tid; col < N; col += blockSize) {
            float val = A[rowStart + col];
            if (val != 0.0f) {
                sum += val * x[col];
            }
        }

        sdata[tid] = sum;
        __syncthreads();

        // Tree reduction down to warp level
        for (int s = blockSize / 2; s > WARP_SIZE; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Final warp reduction using shuffle
        if (tid < WARP_SIZE) {
            float val = sdata[tid];
            if (blockSize >= 64) val += sdata[tid + WARP_SIZE];
            val = warpReduceSum(val);
            if (tid == 0) {
                y[row] = val;
            }
        }
    }
}

/**
 * Kernel 4: Hybrid approach - caches x in shared memory, uses
 * shuffle-based final reduction instead of volatile trick.
 */
__global__ void spmvHybrid(const float* __restrict__ A,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ float shared[];
    float* sdata = shared;
    float* sx = shared + blockSize;

    if (row >= M) return;

    // Cooperatively load x into shared memory
    for (int i = tid; i < N; i += blockSize) {
        sx[i] = x[i];
    }
    __syncthreads();

    float sum = 0.0f;
    int rowStart = row * N;

    for (int col = tid; col < N; col += blockSize) {
        float val = A[rowStart + col];
        if (val != 0.0f) {
            sum += val * sx[col];
        }
    }

    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction down to warp level
    for (int s = blockSize / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp reduction using shuffle
    if (tid < WARP_SIZE) {
        float val = sdata[tid];
        if (blockSize >= 64) val += sdata[tid + WARP_SIZE];
        val = warpReduceSum(val);
        if (tid == 0) {
            y[row] = val;
        }
    }
}

// CPU reference implementation
void spmvCpu(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}
