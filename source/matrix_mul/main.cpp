#include "matrix_mul.h"
#include "cuda_helpers.h"
#include <stdio.h>
#include <cmath>

// Matrix size (4x4)
#define N 4

// Host function to print a matrix
void printMatrix(float *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

// Function to compare CPU and GPU results
int compareResults(float *cpu_C, float *gpu_C, int n, float tolerance = 1e-5) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(cpu_C[i] - gpu_C[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu_C[i], gpu_C[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    // Matrix dimensions
    const int n = N;
    const int size = n * n * sizeof(float);

    // Host matrices
    float h_A[N * N] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float h_B[N * N] = {
        1, 0, 0, 1,
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 0, 0, 1
    };
    float h_C_cpu[N * N], h_C_gpu[N * N];

    // Device matrices
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // CPU matrix multiplication
    double start_cpu = getTime();
    matrixMulCPU(h_A, h_B, h_C_cpu, n);
    double end_cpu = getTime();
    double cpu_time = end_cpu - start_cpu;

    // CUDA matrix multiplication
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    double gpu_time = gpu_time_ms / 1000.0; // Convert to seconds

    // Copy GPU result back to host
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    // Print matrices
    printf("Matrix A:\n");
    printMatrix(h_A, n);
    printf("\nMatrix B:\n");
    printMatrix(h_B, n);
    printf("\nMatrix C (CPU Result):\n");
    printMatrix(h_C_cpu, n);
    printf("\nMatrix C (GPU Result):\n");
    printMatrix(h_C_gpu, n);

    // Compare results
    printf("\nComparing CPU and GPU results...\n");
    if (compareResults(h_C_cpu, h_C_gpu, n)) {
        printf("Results match!\n");
    } else {
        printf("Results do NOT match!\n");
    }

    // Print execution times
    printf("\nExecution Times:\n");
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f seconds\n", gpu_time);
    printf("Speedup (CPU Time / GPU Time): %f\n", cpu_time / gpu_time);

    // Free device memory and CUDA events
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
