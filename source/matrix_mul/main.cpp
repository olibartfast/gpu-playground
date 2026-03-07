#ifdef GPU_OPENCL_BACKEND
#include "opencl/matrix_mul.h"
#else
#include "cuda/matrix_mul.h"
#endif
#include <stdio.h>
#include <chrono>
#include <cmath>

#define N 4

void printMatrix(const float* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%8.2f ", matrix[i * n + j]);
        printf("\n");
    }
}

int compareResults(const float* cpu_C, const float* gpu_C, int n, float tolerance = 1e-5f) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(cpu_C[i] - gpu_C[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu_C[i], gpu_C[i]);
            return 0;
        }
    }
    return 1;
}

int main() {
    const int n = N;

    float h_A[N * N] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };
    float h_B[N * N] = {
        1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1
    };
    float h_C_cpu[N * N], h_C_gpu[N * N];

    auto t0 = std::chrono::steady_clock::now();
    matrixMulCPU(h_A, h_B, h_C_cpu, n);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    auto g0 = std::chrono::steady_clock::now();
    matrixMulGPU(h_A, h_B, h_C_gpu, n);
    auto g1 = std::chrono::steady_clock::now();
    double gpu_time = std::chrono::duration<double>(g1 - g0).count();

    printf("Matrix A:\n"); printMatrix(h_A, n);
    printf("\nMatrix B:\n"); printMatrix(h_B, n);
    printf("\nMatrix C (CPU Result):\n"); printMatrix(h_C_cpu, n);
    printf("\nMatrix C (GPU Result):\n"); printMatrix(h_C_gpu, n);

    printf("\nComparing CPU and GPU results...\n");
    if (compareResults(h_C_cpu, h_C_gpu, n)) printf("Results match!\n");
    else printf("Results do NOT match!\n");

    printf("\nExecution Times:\n");
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f seconds\n", gpu_time);
    if (gpu_time > 0) printf("Speedup (CPU Time / GPU Time): %f\n", cpu_time / gpu_time);
    return 0;
}
