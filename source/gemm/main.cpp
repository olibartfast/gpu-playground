#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/gemm.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/gemm.h"
#else
#include "cuda/gemm.h"
#endif
#include <stdio.h>
#include "benchmark.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows && i < 4; i++) {
        printf("  [");
        for (int j = 0; j < cols && j < 4; j++) printf("%8.2f", matrix[i * cols + j]);
        if (cols > 4) printf(" ...");
        printf(" ]\n");
    }
    if (rows > 4) printf("  ...\n");
    printf("\n");
}

int compareResults(const float* cpu_C, const float* gpu_C, int size, float tolerance = 0.1f) {
    float maxDiff = 0.0f, maxRelDiff = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(cpu_C[i] - gpu_C[i]);
        float rel = diff / (fabsf(cpu_C[i]) + 1e-6f);
        if (diff > maxDiff) maxDiff = diff;
        if (rel > maxRelDiff) maxRelDiff = rel;
        if (diff > tolerance) {
            if (mismatches < 5)
                printf("Mismatch at index %d: CPU = %f, GPU = %f (diff = %f)\n",
                       i, cpu_C[i], gpu_C[i], diff);
            mismatches++;
        }
    }
    printf("Max absolute difference: %e\n", maxDiff);
    printf("Max relative difference: %e\n", maxRelDiff);
    if (mismatches > 5) printf("... and %d more mismatches\n", mismatches - 5);
    return maxDiff < tolerance;
}

int main() {
    printf("=== GEMM (General Matrix Multiplication) Test ===\n\n");

    // Small test: A(2x3) * B(3x2), alpha=1, beta=0 → [[58,64],[139,154]]
    printf("Small Test (2x3 * 3x2):\n");
    printf("----------------------------------------\n");

    const int M1 = 2, K1 = 3, N1 = 2;
    float alpha1 = 1.0f, beta1 = 0.0f;
    float h_A1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float h_B1[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float h_C1_cpu[4] = {}, h_C1_gpu[4] = {};

    printMatrix(h_A1, M1, K1, "Matrix A (2x3)");
    printMatrix(h_B1, K1, N1, "Matrix B (3x2)");
    printf("α = %.1f, β = %.1f\n\n", alpha1, beta1);

    // GEMM reads C for the beta * C term, so each iteration must restart from
    // the same C -- otherwise repeated calls accumulate into their own output.
    float h_C1_seed[4] = {};
    auto reset_C1_cpu = [&] { std::copy(h_C1_seed, h_C1_seed + M1 * N1, h_C1_cpu); };
    auto reset_C1_gpu = [&] { std::copy(h_C1_seed, h_C1_seed + M1 * N1, h_C1_gpu); };

    BenchResult cpu_bench = benchmarkWithReset(
        [&] { gemmCpu(h_A1, h_B1, h_C1_cpu, alpha1, beta1, M1, N1, K1); }, reset_C1_cpu);
    BenchResult gpu_bench = benchmarkWithReset(
        [&] { gemmGPU(h_A1, h_B1, h_C1_gpu, alpha1, beta1, M1, N1, K1); }, reset_C1_gpu);

    printMatrix(h_C1_cpu, M1, N1, "Matrix C (CPU Result)");
    printMatrix(h_C1_gpu, M1, N1, "Matrix C (GPU Result)");
    printf("Expected: [[58, 64], [139, 154]]\n\n");
    printf("Comparing CPU and GPU results...\n");
    bool small_ok = compareResults(h_C1_cpu, h_C1_gpu, M1 * N1);
    printf("%s\n", small_ok ? "Results match!" : "Results do NOT match!");
    printf("\nExecution Times:\n");
    printBench("CPU:", cpu_bench);
    printBench("GPU:", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

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
    float* h_C2_seed = new float[M2 * N2];

    printf("M=%d, K=%d, N=%d\nα = %.1f, β = %.1f\n\n", M2, K2, N2, alpha2, beta2);

    // A single random draw hides boundary bugs, so validate over several seeds.
    // The last trial leaves its data in place for the benchmark below.
    const unsigned seeds[] = {42u, 1337u, 2024u, 7u, 99u};
    const int trials = (int)(sizeof(seeds) / sizeof(seeds[0]));
    bool large_ok = true;
    for (int trial = 0; trial < trials; trial++) {
        srand(seeds[trial]);
        for (int i = 0; i < M2 * K2; i++) h_A2[i] = (float)(rand() % 100) / 50.0f - 1.0f;
        for (int i = 0; i < K2 * N2; i++) h_B2[i] = (float)(rand() % 100) / 50.0f - 1.0f;
        for (int i = 0; i < M2 * N2; i++) {
            float val = (float)(rand() % 100) / 50.0f - 1.0f;
            h_C2_cpu[i] = h_C2_gpu[i] = h_C2_seed[i] = val;
        }

        gemmCpu(h_A2, h_B2, h_C2_cpu, alpha2, beta2, M2, N2, K2);
        gemmGPU(h_A2, h_B2, h_C2_gpu, alpha2, beta2, M2, N2, K2);

        printf("Trial %d/%d (seed %u):\n", trial + 1, trials, seeds[trial]);
        bool ok = compareResults(h_C2_cpu, h_C2_gpu, M2 * N2);
        printf("  -> %s\n\n", ok ? "PASSED" : "FAILED");
        large_ok = large_ok && ok;
    }

    printMatrix(h_C2_cpu, M2, N2, "Matrix C (CPU Result) - First 4x4");
    printMatrix(h_C2_gpu, M2, N2, "Matrix C (GPU Result) - First 4x4");
    printf("Large test: %s\n", large_ok ? "PASSED" : "FAILED");

    // The naive CPU reference costs ~200 ms per call here, so it gets a shorter
    // protocol; the GPU path -- the one worth measuring precisely -- keeps the default.
    cpu_bench = benchmarkWithReset(
        [&] { gemmCpu(h_A2, h_B2, h_C2_cpu, alpha2, beta2, M2, N2, K2); },
        [&] { std::copy(h_C2_seed, h_C2_seed + M2 * N2, h_C2_cpu); }, 1, 3);
    gpu_bench = benchmarkWithReset(
        [&] { gemmGPU(h_A2, h_B2, h_C2_gpu, alpha2, beta2, M2, N2, K2); },
        [&] { std::copy(h_C2_seed, h_C2_seed + M2 * N2, h_C2_gpu); });

    printf("\nExecution Times:\n");
    printBench("CPU:", cpu_bench);
    printBench("GPU:", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

    delete[] h_A2; delete[] h_B2; delete[] h_C2_cpu; delete[] h_C2_gpu; delete[] h_C2_seed;

    bool ok = small_ok && large_ok;
    printf("\nOverall result: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
