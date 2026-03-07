#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/gemm.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/gemm.h"
#else
#include "cuda/gemm.h"
#endif
#include <stdio.h>
#include <chrono>
#include <cmath>
#include <cstdlib>

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

    auto t0 = std::chrono::steady_clock::now();
    gemmCpu(h_A1, h_B1, h_C1_cpu, alpha1, beta1, M1, N1, K1);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();

    auto g0 = std::chrono::steady_clock::now();
    gemmGPU(h_A1, h_B1, h_C1_gpu, alpha1, beta1, M1, N1, K1);
    auto g1 = std::chrono::steady_clock::now();
    double gpu_time = std::chrono::duration<double>(g1 - g0).count();

    printMatrix(h_C1_cpu, M1, N1, "Matrix C (CPU Result)");
    printMatrix(h_C1_gpu, M1, N1, "Matrix C (GPU Result)");
    printf("Expected: [[58, 64], [139, 154]]\n\n");
    printf("Comparing CPU and GPU results...\n");
    if (compareResults(h_C1_cpu, h_C1_gpu, M1 * N1)) printf("Results match!\n");
    else printf("Results do NOT match!\n");
    printf("\nExecution Times:\nCPU Time: %f seconds\nGPU Time: %f seconds\n", cpu_time, gpu_time);
    if (gpu_time > 0) printf("Speedup (CPU Time / GPU Time): %f\n", cpu_time / gpu_time);

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

    srand(42);
    for (int i = 0; i < M2 * K2; i++) h_A2[i] = (float)(rand() % 100) / 50.0f - 1.0f;
    for (int i = 0; i < K2 * N2; i++) h_B2[i] = (float)(rand() % 100) / 50.0f - 1.0f;
    for (int i = 0; i < M2 * N2; i++) {
        float val = (float)(rand() % 100) / 50.0f - 1.0f;
        h_C2_cpu[i] = h_C2_gpu[i] = val;
    }
    printf("M=%d, K=%d, N=%d\nα = %.1f, β = %.1f\n\n", M2, K2, N2, alpha2, beta2);

    t0 = std::chrono::steady_clock::now();
    gemmCpu(h_A2, h_B2, h_C2_cpu, alpha2, beta2, M2, N2, K2);
    t1 = std::chrono::steady_clock::now();
    cpu_time = std::chrono::duration<double>(t1 - t0).count();

    g0 = std::chrono::steady_clock::now();
    gemmGPU(h_A2, h_B2, h_C2_gpu, alpha2, beta2, M2, N2, K2);
    g1 = std::chrono::steady_clock::now();
    gpu_time = std::chrono::duration<double>(g1 - g0).count();

    printMatrix(h_C2_cpu, M2, N2, "Matrix C (CPU Result) - First 4x4");
    printMatrix(h_C2_gpu, M2, N2, "Matrix C (GPU Result) - First 4x4");
    printf("Comparing CPU and GPU results...\n");
    if (compareResults(h_C2_cpu, h_C2_gpu, M2 * N2)) printf("Test PASSED!\n");
    else printf("Test FAILED!\n");
    printf("\nExecution Times:\nCPU Time: %f seconds\nGPU Time: %f seconds\n", cpu_time, gpu_time);
    printf("Speedup (CPU Time / GPU Time): %.2fx\n", cpu_time / gpu_time);

    delete[] h_A2; delete[] h_B2; delete[] h_C2_cpu; delete[] h_C2_gpu;
    return 0;
}
