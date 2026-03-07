#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/prefix_sum.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/prefix_sum.h"
#else
#include "cuda/prefix_sum.h"
#endif
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>

#define PRINT

void print(const float* input, int N, const std::string& message = "") {
    if (!message.empty()) std::cout << message << ": ";
    for (int i = 0; i < N; i++) std::cout << input[i] << " ";
    std::cout << std::endl;
}

int main() {
    int N = 4;
    float* input = (float*)malloc(sizeof(float) * N);
    float* output_cpu = (float*)malloc(sizeof(float) * N);
    float* output_gpu = (float*)malloc(sizeof(float) * N);
    if (!input || !output_cpu || !output_gpu) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    float test_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < N; i++) { input[i] = test_input[i]; output_cpu[i] = output_gpu[i] = 0.0f; }
    #ifdef PRINT
    print(input, N, "Starting list");
    #endif

    auto start = std::chrono::steady_clock::now();
    prefix_scan_cpu(input, output_cpu, N);
    auto end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print(output_cpu, N, "CPU prefix scan");
    #endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();
    prefix_scan_gpu(input, output_gpu, N);
    end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print(output_gpu, N, "GPU prefix scan");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    float expected[] = {1.0f, 3.0f, 6.0f, 10.0f};
    bool passed = true;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabs(output_gpu[i] - expected[i]);
        if (diff > 1e-5) { passed = false; max_diff = fmax(max_diff, diff); }
    }
    if (passed) std::cout << "Test passed!" << std::endl;
    else std::cout << "Test failed! Max difference: " << max_diff << std::endl;

    free(input);
    free(output_cpu);
    free(output_gpu);
    return passed ? 0 : 1;
}
