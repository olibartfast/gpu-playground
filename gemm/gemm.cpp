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
 * Precision: FP16 storage, FP32 accumulation
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <sys/time.h>
#include <cmath>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Function to get current time in seconds
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Tile dimensions for shared memory blocking
#define TILE_SIZE 16

/**
 * Kernel 1: Basic Tiled GEMM
 * 
 * Each thread block computes a TILE_SIZE x TILE_SIZE tile of C.
 * Uses shared memory to cache tiles of A and B for reuse.
 * 
 * Pattern: Tiled Partitioning + Map + Reduction
 */
__global__ void gemmTiled(const half* A, const half* B, half* C,
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
            As[ty][tx] = __half2float(A[row * K + aCol]);
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = __half2float(B[bRow * N + col]);
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
        float c_val = __half2float(C[row * N + col]);
        float result = alpha * sum + beta * c_val;
        C[row * N + col] = __float2half(result);
    }
}

/**
 * Kernel 2: Tiled GEMM with 2x2 thread tiling (better arithmetic intensity)
 * 
 * Each thread computes a 2x2 block of output elements.
 * Increases arithmetic intensity and register usage.
 */
#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define THREAD_TILE 2

__global__ void gemmTiled2x2(const half* A, const half* B, half* C,
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
                As[tx][ty * THREAD_TILE + i] = __half2float(A[row * K + col]);
            } else {
                As[tx][ty * THREAD_TILE + i] = 0.0f;
            }
        }
        
        // Cooperative loading of B tile
        for (int i = 0; i < THREAD_TILE; i++) {
            int row = t * TILE_K + ty;
            int col = colBase + i;
            if (row < K && col < N) {
                Bs[ty][tx * THREAD_TILE + i] = __half2float(B[row * N + col]);
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
                float c_val = __half2float(C[row * N + col]);
                float result = alpha * sum[i][j] + beta * c_val;
                C[row * N + col] = __float2half(result);
            }
        }
    }
}

/**
 * Kernel 3: Optimized with vectorized loads and better memory coalescing
 */
__global__ void gemmOptimized(const half* A, const half* B, half* C,
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
    const half* A_row = A + row * K;
    const half* B_col = B + col;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int tileOffset = t * TILE_SIZE;
        
        // Coalesced load of A (row-major, consecutive threads read consecutive elements)
        int aIdx = tileOffset + tx;
        As[ty][tx] = (row < M && aIdx < K) ? __half2float(A_row[aIdx]) : 0.0f;
        
        // Coalesced load of B
        int bIdx = (tileOffset + ty) * N;
        Bs[ty][tx] = (tileOffset + ty < K && col < N) ? __half2float(B_col[bIdx]) : 0.0f;
        
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
        float c_val = __half2float(C[row * N + col]);
        C[row * N + col] = __float2half(fmaf(alpha, sum, beta * c_val));
    }
}

// CPU matrix multiplication reference
void gemmCpu(const half* A, const half* B, half* C,
             float alpha, float beta,
             int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            float c_val = __half2float(C[i * N + j]);
            C[i * N + j] = __float2half(alpha * sum + beta * c_val);
        }
    }
}

// Function to print a matrix
void printMatrix(const half* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows && i < 4; i++) {
        printf("  [");
        for (int j = 0; j < cols && j < 4; j++) {
            printf("%8.2f", __half2float(matrix[i * cols + j]));
        }
        if (cols > 4) printf(" ...");
        printf(" ]\n");
    }
    if (rows > 4) printf("  ...\n");
    printf("\n");
}

// Function to compare CPU and GPU results
int compareResults(const half* cpu_C, const half* gpu_C, int size, float tolerance = 0.1f) {
    float maxDiff = 0.0f;
    float maxRelDiff = 0.0f;
    int mismatches = 0;
    
    for (int i = 0; i < size; i++) {
        float cpu_val = __half2float(cpu_C[i]);
        float gpu_val = __half2float(gpu_C[i]);
        float diff = fabsf(cpu_val - gpu_val);
        float rel = diff / (fabsf(cpu_val) + 1e-6f);
        
        if (diff > maxDiff) maxDiff = diff;
        if (rel > maxRelDiff) maxRelDiff = rel;
        
        if (diff > tolerance) {
            if (mismatches < 5) {  // Print first 5 mismatches
                printf("Mismatch at index %d: CPU = %f, GPU = %f (diff = %f)\n", 
                       i, cpu_val, gpu_val, diff);
            }
            mismatches++;
        }
    }
    
    printf("Max absolute difference: %e\n", maxDiff);
    printf("Max relative difference: %e\n", maxRelDiff);
    
    if (mismatches > 5) {
        printf("... and %d more mismatches\n", mismatches - 5);
    }
    
    return maxDiff < tolerance;
}

int main() {
    printf("=== GEMM (General Matrix Multiplication) Test ===\n\n");
    
    // Small test: Example from problem statement
    // A = [[1,2,3], [4,5,6]]  (2x3)
    // B = [[7,8], [9,10], [11,12]]  (3x2)
    // C_init = [[0,0], [0,0]]  (2x2)
    // α = 1.0, β = 0.0
    // Result: [[58, 64], [139, 154]]
    
    printf("Small Test (2x3 * 3x2):\n");
    printf("----------------------------------------\n");
    
    const int M1 = 2, K1 = 3, N1 = 2;
    float alpha1 = 1.0f, beta1 = 0.0f;
    
    half h_A1[] = {__float2half(1), __float2half(2), __float2half(3),
                   __float2half(4), __float2half(5), __float2half(6)};
    half h_B1[] = {__float2half(7), __float2half(8),
                   __float2half(9), __float2half(10),
                   __float2half(11), __float2half(12)};
    half h_C1_cpu[4], h_C1_gpu[4];
    
    // Initialize C matrices to 0
    for (int i = 0; i < 4; i++) {
        h_C1_cpu[i] = __float2half(0);
        h_C1_gpu[i] = __float2half(0);
    }
    
    printMatrix(h_A1, M1, K1, "Matrix A (2x3)");
    printMatrix(h_B1, K1, N1, "Matrix B (3x2)");
    printf("α = %.1f, β = %.1f\n\n", alpha1, beta1);
    
    // CPU computation
    double start_cpu = getTime();
    gemmCpu(h_A1, h_B1, h_C1_cpu, alpha1, beta1, M1, N1, K1);
    double end_cpu = getTime();
    double cpu_time = end_cpu - start_cpu;
    
    // GPU computation
    half *d_A1, *d_B1, *d_C1;
    CUDA_CHECK(cudaMalloc(&d_A1, M1 * K1 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B1, K1 * N1 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C1, M1 * N1 * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_A1, h_A1, M1 * K1 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, h_B1, K1 * N1 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C1, h_C1_gpu, M1 * N1 * sizeof(half), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    dim3 blockDim1(TILE_SIZE, TILE_SIZE);
    dim3 gridDim1((N1 + TILE_SIZE - 1) / TILE_SIZE, (M1 + TILE_SIZE - 1) / TILE_SIZE);
    
    CUDA_CHECK(cudaEventRecord(start));
    gemmTiled<<<gridDim1, blockDim1>>>(d_A1, d_B1, d_C1, alpha1, beta1, M1, N1, K1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    double gpu_time = gpu_time_ms / 1000.0;
    
    CUDA_CHECK(cudaMemcpy(h_C1_gpu, d_C1, M1 * N1 * sizeof(half), cudaMemcpyDeviceToHost));
    
    printMatrix(h_C1_cpu, M1, N1, "Matrix C (CPU Result)");
    printMatrix(h_C1_gpu, M1, N1, "Matrix C (GPU Result)");
    printf("Expected: [[58, 64], [139, 154]]\n\n");
    
    printf("Comparing CPU and GPU results...\n");
    if (compareResults(h_C1_cpu, h_C1_gpu, M1 * N1)) {
        printf("Results match!\n");
    } else {
        printf("Results do NOT match!\n");
    }
    
    printf("\nExecution Times:\n");
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f seconds\n", gpu_time);
    if (gpu_time > 0) {
        printf("Speedup (CPU Time / GPU Time): %f\n", cpu_time / gpu_time);
    }
    
    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_C1));
    
    printf("\n========================================\n\n");
    
    // Large test
    printf("Large Test (512x256 * 256x512):\n");
    printf("----------------------------------------\n");
    
    const int M2 = 512, K2 = 256, N2 = 512;
    float alpha2 = 1.5f, beta2 = 0.5f;
    
    half* h_A2 = new half[M2 * K2];
    half* h_B2 = new half[K2 * N2];
    half* h_C2_cpu = new half[M2 * N2];
    half* h_C2_gpu = new half[M2 * N2];
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < M2 * K2; i++) {
        h_A2[i] = __float2half((float)(rand() % 100) / 50.0f - 1.0f);
    }
    for (int i = 0; i < K2 * N2; i++) {
        h_B2[i] = __float2half((float)(rand() % 100) / 50.0f - 1.0f);
    }
    for (int i = 0; i < M2 * N2; i++) {
        float val = (float)(rand() % 100) / 50.0f - 1.0f;
        h_C2_cpu[i] = __float2half(val);
        h_C2_gpu[i] = __float2half(val);
    }
    
    printf("M=%d, K=%d, N=%d\n", M2, K2, N2);
    printf("α = %.1f, β = %.1f\n\n", alpha2, beta2);
    
    // CPU computation
    start_cpu = getTime();
    gemmCpu(h_A2, h_B2, h_C2_cpu, alpha2, beta2, M2, N2, K2);
    end_cpu = getTime();
    cpu_time = end_cpu - start_cpu;
    
    // GPU computation
    half *d_A2, *d_B2, *d_C2;
    CUDA_CHECK(cudaMalloc(&d_A2, M2 * K2 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B2, K2 * N2 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C2, M2 * N2 * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_A2, h_A2, M2 * K2 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2, h_B2, K2 * N2 * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C2, h_C2_gpu, M2 * N2 * sizeof(half), cudaMemcpyHostToDevice));
    
    dim3 blockDim2(TILE_SIZE, TILE_SIZE);
    dim3 gridDim2((N2 + TILE_SIZE - 1) / TILE_SIZE, (M2 + TILE_SIZE - 1) / TILE_SIZE);
    
    CUDA_CHECK(cudaEventRecord(start));
    gemmOptimized<<<gridDim2, blockDim2>>>(d_A2, d_B2, d_C2, alpha2, beta2, M2, N2, K2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    gpu_time = gpu_time_ms / 1000.0;
    
    CUDA_CHECK(cudaMemcpy(h_C2_gpu, d_C2, M2 * N2 * sizeof(half), cudaMemcpyDeviceToHost));
    
    printMatrix(h_C2_cpu, M2, N2, "Matrix C (CPU Result) - First 4x4");
    printMatrix(h_C2_gpu, M2, N2, "Matrix C (GPU Result) - First 4x4");
    
    printf("Comparing CPU and GPU results...\n");
    if (compareResults(h_C2_cpu, h_C2_gpu, M2 * N2)) {
        printf("Test PASSED!\n");
    } else {
        printf("Test FAILED!\n");
    }
    
    printf("\nExecution Times:\n");
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f seconds\n", gpu_time);
    printf("Speedup (CPU Time / GPU Time): %.2fx\n", cpu_time / gpu_time);
    
    // Cleanup
    delete[] h_A2;
    delete[] h_B2;
    delete[] h_C2_cpu;
    delete[] h_C2_gpu;
    
    CUDA_CHECK(cudaFree(d_A2));
    CUDA_CHECK(cudaFree(d_B2));
    CUDA_CHECK(cudaFree(d_C2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
