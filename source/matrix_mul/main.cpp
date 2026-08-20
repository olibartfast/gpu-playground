#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/matrix_mul.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/matrix_mul.h"
#else
#include "cuda/matrix_mul.h"
#endif
#include <stdio.h>
#include "benchmark.h"
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

    BenchResult cpu_bench = benchmark([&] { matrixMulCPU(h_A, h_B, h_C_cpu, n); });
    BenchResult gpu_bench = benchmark([&] { matrixMulGPU(h_A, h_B, h_C_gpu, n); });

    printf("Matrix A:\n"); printMatrix(h_A, n);
    printf("\nMatrix B:\n"); printMatrix(h_B, n);
    printf("\nMatrix C (CPU Result):\n"); printMatrix(h_C_cpu, n);
    printf("\nMatrix C (GPU Result):\n"); printMatrix(h_C_gpu, n);

    printf("\nComparing CPU and GPU results...\n");
    if (compareResults(h_C_cpu, h_C_gpu, n)) printf("Results match!\n");
    else printf("Results do NOT match!\n");

    printf("\nExecution Times:\n");
    printBench("CPU:", cpu_bench);
    printBench("GPU:", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);
    return 0;
}
