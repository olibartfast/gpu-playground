#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/swiglu.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/swiglu.h"
#else
#include "cuda/swiglu.h"
#endif
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <vector>

#define PRINT

void print(const float* input, int N, const std::string& message = "") {
    if (!message.empty()) {
        std::cout << message << ": ";
    }
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < N; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    int N = 8;

    std::vector<float> input(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_gpu(N);

    if (N == 8) {
        float sample_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
        for (int i = 0; i < N; i++) input[i] = sample_values[i];
    } else {
        for (int i = 0; i < N; i++) input[i] = (float)(i % 100);
    }

    #ifdef PRINT
    print(input.data(), N, "Input");
    #endif

    auto start_cpu = std::chrono::steady_clock::now();
    swiglu_cpu(input.data(), output_cpu.data(), N);
    auto end_cpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_cpu.data(), N/2, "CPU SwiGLU");
    #endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count()
              << " ms" << std::endl << std::endl;

    auto start_gpu = std::chrono::steady_clock::now();
    swiglu_gpu(input.data(), output_gpu.data(), N);
    auto end_gpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_gpu.data(), N/2, "GPU SwiGLU");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count()
              << " ms" << std::endl << std::endl;

    bool results_match = true;
    for (int i = 0; i < N/2; i++) {
        if (fabsf(output_cpu[i] - output_gpu[i]) > 1e-5f) {
            results_match = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << output_cpu[i]
                      << " GPU=" << output_gpu[i] << std::endl;
            break;
        }
    }
    std::cout << (results_match ? "Results match!" : "Results do not match!") << std::endl;
    return 0;
}
