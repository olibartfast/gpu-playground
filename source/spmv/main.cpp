#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/spmv.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/spmv.h"
#else
#include "cuda/spmv.h"
#endif
#include <cstdio>
#include "benchmark.h"
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

    BenchResult cpu_bench = benchmark([&] { spmvCpu(h_A1, h_x1, h_y1_cpu, M1, N1); });
    BenchResult gpu_bench = benchmark([&] { spmvGPU(h_A1, h_x1, h_y1_gpu, M1, N1); });

    printVector(h_y1_cpu, M1, "Vector y (CPU Result)");
    printVector(h_y1_gpu, M1, "Vector y (GPU Result)");
    printf("Expected: [7.00, 6.00, 19.00]\n\n");
    printf("Comparing CPU and GPU results...\n");
    bool small_ok = compareResults(h_y1_cpu, h_y1_gpu, M1);
    printf("%s\n", small_ok ? "Results match!" : "Results do NOT match!");
    printf("\n");
    printBench("CPU:", cpu_bench);
    printBench("GPU:", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

    printf("\n========================================\n\n");

    // --- Large Test ---
    printf("Large Test (1000x1000 sparse matrix):\n");
    printf("----------------------------------------\n");

    constexpr int M2 = 1000, N2 = 1000;
    float* h_A2 = new float[M2 * N2];
    float* h_x2 = new float[N2];
    float* h_y2_cpu = new float[M2];
    float* h_y2_gpu = new float[M2];

    // A single random draw hides boundary bugs, so validate over several seeds.
    // The last trial leaves its matrix in place for the benchmark below.
    const unsigned seeds[] = {42u, 1337u, 2024u, 7u, 99u};
    const int trials = (int)(sizeof(seeds) / sizeof(seeds[0]));
    bool large_ok = true;
    int nnz2 = 0;
    for (int trial = 0; trial < trials; trial++) {
        srand(seeds[trial]);
        nnz2 = 0;
        for (int i = 0; i < M2 * N2; i++) {
            if (rand() % 100 < 35) {
                h_A2[i] = static_cast<float>(rand() % 100) / 10.0f;
                nnz2++;
            } else {
                h_A2[i] = 0.0f;
            }
        }
        for (int i = 0; i < N2; i++) h_x2[i] = static_cast<float>(rand() % 100) / 10.0f;

        spmvCpu(h_A2, h_x2, h_y2_cpu, M2, N2);
        spmvGPU(h_A2, h_x2, h_y2_gpu, M2, N2);

        bool ok = compareResults(h_y2_cpu, h_y2_gpu, M2);
        printf("Trial %d/%d (seed %5u, nnz=%d, %.1f%% sparse): %s\n",
               trial + 1, trials, seeds[trial], nnz2,
               100.0f * (1.0f - static_cast<float>(nnz2) / (M2 * N2)),
               ok ? "PASSED" : "FAILED");
        large_ok = large_ok && ok;
    }
    printf("\n");

    cpu_bench = benchmark([&] { spmvCpu(h_A2, h_x2, h_y2_cpu, M2, N2); });
    gpu_bench = benchmark([&] { spmvGPU(h_A2, h_x2, h_y2_gpu, M2, N2); });

    printVector(h_y2_cpu, M2, "Vector y (CPU) - First 10");
    printVector(h_y2_gpu, M2, "Vector y (GPU) - First 10");
    printf("\nLarge test: %s\n\n", large_ok ? "PASSED" : "FAILED");
    printBench("CPU:", cpu_bench);
    printBench("GPU:", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

    delete[] h_A2;
    delete[] h_x2;
    delete[] h_y2_cpu;
    delete[] h_y2_gpu;

    bool ok = small_ok && large_ok;
    printf("\nOverall result: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
