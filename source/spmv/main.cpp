#ifdef GPU_OPENCL_BACKEND
#include "opencl/spmv.h"
#else
#include "cuda/spmv.h"
#endif
#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>

void printVector(const float* v, int n, const char* name) {
    printf("%s: [", name);
    int limit = std::min(n, 10);
    for (int i = 0; i < limit; i++) printf("%.2f%s", v[i], i < limit - 1 ? ", " : "");
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
            if (mismatches < 5)
                printf("Mismatch at index %d: CPU = %f, GPU = %f (diff = %f)\n",
                       i, cpu_y[i], gpu_y[i], diff);
            mismatches++;
        }
    }
    printf("Max difference: %e\n", maxDiff);
    if (mismatches > 5) printf("... and %d more mismatches\n", mismatches - 5);
    return maxDiff < tolerance;
}

int main() {
    printf("=== Sparse Matrix-Vector Multiplication (SpMV) Test ===\n\n");

    // --- Small Test ---
    printf("Small Test (3x3 sparse matrix):\n");
    printf("----------------------------------------\n");

    constexpr int M1 = 3, N1 = 3;
    float h_A1[] = {1, 0, 2, 0, 3, 0, 4, 0, 5};
    float h_x1[] = {1, 2, 3};
    float h_y1_cpu[3] = {}, h_y1_gpu[3] = {};

    int nnz1 = 5;
    printf("Matrix A:\n");
    for (int i = 0; i < M1; i++) {
        printf("  [");
        for (int j = 0; j < N1; j++) printf("%5.1f ", h_A1[i * N1 + j]);
        printf("]\n");
    }
    printVector(h_x1, N1, "\nVector x");
    printf("Non-zero elements: %d (%.1f%% sparse)\n\n", nnz1,
           100.0f * (1.0f - static_cast<float>(nnz1) / (M1 * N1)));

    auto t0 = std::chrono::steady_clock::now();
    spmvCpu(h_A1, h_x1, h_y1_cpu, M1, N1);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    auto g0 = std::chrono::steady_clock::now();
    spmvGPU(h_A1, h_x1, h_y1_gpu, M1, N1);
    auto g1 = std::chrono::steady_clock::now();
    double gpu_time = std::chrono::duration<double>(g1 - g0).count();

    printVector(h_y1_cpu, M1, "Vector y (CPU Result)");
    printVector(h_y1_gpu, M1, "Vector y (GPU Result)");
    printf("Expected: [7.00, 6.00, 19.00]\n\n");
    printf("Comparing CPU and GPU results...\n");
    printf("%s\n", compareResults(h_y1_cpu, h_y1_gpu, M1) ? "Results match!" : "Results do NOT match!");
    printf("\nCPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f seconds\n", gpu_time);
    if (gpu_time > 0) printf("Speedup: %.2fx\n", cpu_time / gpu_time);

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
    for (int i = 0; i < N2; i++) h_x2[i] = static_cast<float>(rand() % 100) / 10.0f;
    printf("M=%d, N=%d, nnz=%d (%.1f%% sparse)\n\n",
           M2, N2, nnz2, 100.0f * (1.0f - static_cast<float>(nnz2) / (M2 * N2)));

    t0 = std::chrono::steady_clock::now();
    spmvCpu(h_A2, h_x2, h_y2_cpu, M2, N2);
    t1 = std::chrono::steady_clock::now();
    cpu_time = std::chrono::duration<double>(t1 - t0).count();

    g0 = std::chrono::steady_clock::now();
    spmvGPU(h_A2, h_x2, h_y2_gpu, M2, N2);
    g1 = std::chrono::steady_clock::now();
    gpu_time = std::chrono::duration<double>(g1 - g0).count();

    printVector(h_y2_cpu, M2, "Vector y (CPU) - First 10");
    printVector(h_y2_gpu, M2, "Vector y (GPU) - First 10");
    printf("\nComparing CPU and GPU results...\n");
    printf("Test %s!\n", compareResults(h_y2_cpu, h_y2_gpu, M2) ? "PASSED" : "FAILED");
    printf("\nCPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f seconds\n", gpu_time);
    if (gpu_time > 0) printf("Speedup: %.2fx\n", cpu_time / gpu_time);

    delete[] h_A2;
    delete[] h_x2;
    delete[] h_y2_cpu;
    delete[] h_y2_gpu;
    return 0;
}
