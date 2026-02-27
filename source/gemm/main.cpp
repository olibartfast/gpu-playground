#include "gemm.h"
#include "cuda_helpers.h"
#include <stdio.h>
#include <cmath>
#include <cstdlib>

// Function to print a matrix
void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows && i < 4; i++) {
        printf("  [");
        for (int j = 0; j < cols && j < 4; j++) {
            printf("%8.2f", matrix[i * cols + j]);
        }
        if (cols > 4) printf(" ...");
        printf(" ]\n");
    }
    if (rows > 4) printf("  ...\n");
    printf("\n");
}

// Function to compare CPU and GPU results
int compareResults(const float* cpu_C, const float* gpu_C, int size, float tolerance = 0.1f) {
    float maxDiff = 0.0f;
    float maxRelDiff = 0.0f;
    int mismatches = 0;

    for (int i = 0; i < size; i++) {
        float cpu_val = cpu_C[i];
        float gpu_val = gpu_C[i];
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

    float h_A1[] = {1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f};
    float h_B1[] = {7.0f, 8.0f,
                    9.0f, 10.0f,
                    11.0f, 12.0f};
    float h_C1_cpu[4], h_C1_gpu[4];

    // Initialize C matrices to 0
    for (int i = 0; i < 4; i++) {
        h_C1_cpu[i] = 0.0f;
        h_C1_gpu[i] = 0.0f;
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
    float *d_A1, *d_B1, *d_C1;
    CUDA_CHECK(cudaMalloc(&d_A1, M1 * K1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B1, K1 * N1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C1, M1 * N1 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A1, h_A1, M1 * K1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, h_B1, K1 * N1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C1, h_C1_gpu, M1 * N1 * sizeof(float), cudaMemcpyHostToDevice));

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

    CUDA_CHECK(cudaMemcpy(h_C1_gpu, d_C1, M1 * N1 * sizeof(float), cudaMemcpyDeviceToHost));

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

    float* h_A2 = new float[M2 * K2];
    float* h_B2 = new float[K2 * N2];
    float* h_C2_cpu = new float[M2 * N2];
    float* h_C2_gpu = new float[M2 * N2];

    // Initialize with random values
    srand(42);
    for (int i = 0; i < M2 * K2; i++) {
        h_A2[i] = (float)(rand() % 100) / 50.0f - 1.0f;
    }
    for (int i = 0; i < K2 * N2; i++) {
        h_B2[i] = (float)(rand() % 100) / 50.0f - 1.0f;
    }
    for (int i = 0; i < M2 * N2; i++) {
        float val = (float)(rand() % 100) / 50.0f - 1.0f;
        h_C2_cpu[i] = val;
        h_C2_gpu[i] = val;
    }

    printf("M=%d, K=%d, N=%d\n", M2, K2, N2);
    printf("α = %.1f, β = %.1f\n\n", alpha2, beta2);

    // CPU computation
    start_cpu = getTime();
    gemmCpu(h_A2, h_B2, h_C2_cpu, alpha2, beta2, M2, N2, K2);
    end_cpu = getTime();
    cpu_time = end_cpu - start_cpu;

    // GPU computation
    float *d_A2, *d_B2, *d_C2;
    CUDA_CHECK(cudaMalloc(&d_A2, M2 * K2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B2, K2 * N2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C2, M2 * N2 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A2, h_A2, M2 * K2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2, h_B2, K2 * N2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C2, h_C2_gpu, M2 * N2 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim2(TILE_SIZE, TILE_SIZE);
    dim3 gridDim2((N2 + TILE_SIZE - 1) / TILE_SIZE, (M2 + TILE_SIZE - 1) / TILE_SIZE);

    CUDA_CHECK(cudaEventRecord(start));
    gemmOptimized<<<gridDim2, blockDim2>>>(d_A2, d_B2, d_C2, alpha2, beta2, M2, N2, K2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    gpu_time = gpu_time_ms / 1000.0;

    CUDA_CHECK(cudaMemcpy(h_C2_gpu, d_C2, M2 * N2 * sizeof(float), cudaMemcpyDeviceToHost));

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
