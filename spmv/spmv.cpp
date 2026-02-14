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

#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

constexpr unsigned FULL_WARP_MASK = 0xFFFFFFFFu;
constexpr int WARP_SIZE = 32;

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

void printVector(const float* v, int n, const char* name) {
    printf("%s: [", name);
    int limit = std::min(n, 10);
    for (int i = 0; i < limit; i++) {
        printf("%.2f%s", v[i], i < limit - 1 ? ", " : "");
    }
    if (n > 10) printf(", ...");
    printf("]\n");
}

int compareResults(const float* cpu_y, const float* gpu_y, int M, float tolerance = 1e-2f) {
    float maxDiff = 0.0f;
    int mismatches = 0;
    
    for (int i = 0; i < M; i++) {
        float diff = std::abs(cpu_y[i] - gpu_y[i]);
        if (diff > maxDiff) maxDiff = diff;
        
        if (diff > tolerance) {
            if (mismatches < 5) {
                printf("Mismatch at index %d: CPU = %f, GPU = %f (diff = %f)\n", 
                       i, cpu_y[i], gpu_y[i], diff);
            }
            mismatches++;
        }
    }
    
    printf("Max difference: %e\n", maxDiff);
    if (mismatches > 5) {
        printf("... and %d more mismatches\n", mismatches - 5);
    }
    
    return maxDiff < tolerance;
}

int main() {
    printf("=== Sparse Matrix-Vector Multiplication (SpMV) Test ===\n\n");
    
    // --- Small Test ---
    printf("Small Test (3x3 sparse matrix):\n");
    printf("----------------------------------------\n");
    
    constexpr int M1 = 3, N1 = 3;
    float h_A1[] = {1, 0, 2,
                    0, 3, 0,
                    4, 0, 5};
    float h_x1[] = {1, 2, 3};
    float h_y1_cpu[3], h_y1_gpu[3];
    
    int nnz1 = 5;
    printf("Matrix A:\n");
    for (int i = 0; i < M1; i++) {
        printf("  [");
        for (int j = 0; j < N1; j++) {
            printf("%5.1f ", h_A1[i * N1 + j]);
        }
        printf("]\n");
    }
    printVector(h_x1, N1, "\nVector x");
    printf("Non-zero elements: %d (%.1f%% sparse)\n\n", nnz1, 
           100.0f * (1.0f - static_cast<float>(nnz1) / (M1 * N1)));
    
    auto t0 = std::chrono::steady_clock::now();
    spmvCpu(h_A1, h_x1, h_y1_cpu, M1, N1);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();
    
    float *d_A1, *d_x1, *d_y1;
    CUDA_CHECK(cudaMalloc(&d_A1, M1 * N1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x1, N1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y1, M1 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A1, h_A1, M1 * N1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x1, h_x1, N1 * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    
    int blockSize = 256;
    int gridSize = (M1 + blockSize - 1) / blockSize;
    
    CUDA_CHECK(cudaEventRecord(ev_start));
    spmvRowPerThread<<<gridSize, blockSize>>>(d_A1, d_x1, d_y1, M1, N1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    
    float gpu_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, ev_start, ev_stop));
    double gpu_time = gpu_time_ms / 1000.0;
    
    CUDA_CHECK(cudaMemcpy(h_y1_gpu, d_y1, M1 * sizeof(float), cudaMemcpyDeviceToHost));
    
    printVector(h_y1_cpu, M1, "Vector y (CPU Result)");
    printVector(h_y1_gpu, M1, "Vector y (GPU Result)");
    printf("Expected: [7.00, 6.00, 19.00]\n\n");
    
    printf("Comparing CPU and GPU results...\n");
    printf("%s\n", compareResults(h_y1_cpu, h_y1_gpu, M1) ? "Results match!" : "Results do NOT match!");
    
    printf("\nCPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f seconds\n", gpu_time);
    if (gpu_time > 0) printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    
    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_x1));
    CUDA_CHECK(cudaFree(d_y1));
    
    printf("\n========================================\n\n");
    
    // --- Large Test ---
    printf("Large Test (1000x1000 sparse matrix):\n");
    printf("----------------------------------------\n");
    
    constexpr int M2 = 1000, N2 = 1000;
    float* h_A2 = new float[M2 * N2];
    float* h_x2 = new float[N2];
    float* h_y2_cpu = new float[M2];
    float* h_y2_gpu = new float[M2];
    
    srand(42);
    int nnz2 = 0;
    for (int i = 0; i < M2 * N2; i++) {
        if (rand() % 100 < 35) {
            h_A2[i] = static_cast<float>(rand() % 100) / 10.0f;
            nnz2++;
        } else {
            h_A2[i] = 0.0f;
        }
    }
    for (int i = 0; i < N2; i++) {
        h_x2[i] = static_cast<float>(rand() % 100) / 10.0f;
    }
    
    printf("M=%d, N=%d, nnz=%d (%.1f%% sparse)\n\n", 
           M2, N2, nnz2, 100.0f * (1.0f - static_cast<float>(nnz2) / (M2 * N2)));
    
    t0 = std::chrono::steady_clock::now();
    spmvCpu(h_A2, h_x2, h_y2_cpu, M2, N2);
    t1 = std::chrono::steady_clock::now();
    cpu_time = std::chrono::duration<double>(t1 - t0).count();
    
    float *d_A2, *d_x2, *d_y2;
    CUDA_CHECK(cudaMalloc(&d_A2, M2 * N2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x2, N2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y2, M2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A2, h_A2, M2 * N2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, h_x2, N2 * sizeof(float), cudaMemcpyHostToDevice));
    
    int warpsPerBlock = 8;
    blockSize = warpsPerBlock * WARP_SIZE;
    gridSize = (M2 + warpsPerBlock - 1) / warpsPerBlock;
    
    CUDA_CHECK(cudaEventRecord(ev_start));
    spmvWarpPerRow<<<gridSize, blockSize>>>(d_A2, d_x2, d_y2, M2, N2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, ev_start, ev_stop));
    gpu_time = gpu_time_ms / 1000.0;
    
    CUDA_CHECK(cudaMemcpy(h_y2_gpu, d_y2, M2 * sizeof(float), cudaMemcpyDeviceToHost));
    
    printVector(h_y2_cpu, M2, "Vector y (CPU) - First 10");
    printVector(h_y2_gpu, M2, "Vector y (GPU) - First 10");
    
    printf("\nComparing CPU and GPU results...\n");
    printf("Test %s!\n", compareResults(h_y2_cpu, h_y2_gpu, M2) ? "PASSED" : "FAILED");
    
    printf("\nCPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    
    delete[] h_A2;
    delete[] h_x2;
    delete[] h_y2_cpu;
    delete[] h_y2_gpu;
    
    CUDA_CHECK(cudaFree(d_A2));
    CUDA_CHECK(cudaFree(d_x2));
    CUDA_CHECK(cudaFree(d_y2));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    
    return 0;
}