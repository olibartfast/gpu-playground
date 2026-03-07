#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/interleave.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/interleave.h"
#else
#include "cuda/interleave.h"
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

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> output_cpu(2 * N);
    std::vector<float> output_gpu(2 * N);

    if (N == 8) {
        float sample_A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float sample_B[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
        for (int i = 0; i < N; i++) { A[i] = sample_A[i]; B[i] = sample_B[i]; }
    } else {
        for (int i = 0; i < N; i++) {
            A[i] = (float)(i % 100);
            B[i] = (float)((i % 100) + 100);
        }
    }

    #ifdef PRINT
    print(A.data(), N, "Array A");
    print(B.data(), N, "Array B");
    #endif

    auto start_cpu = std::chrono::steady_clock::now();
    interleave_cpu(A.data(), B.data(), output_cpu.data(), N);
    auto end_cpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_cpu.data(), 2 * N, "CPU Interleave");
    #endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count()
              << " ms" << std::endl << std::endl;

    auto start_gpu = std::chrono::steady_clock::now();
    interleave_gpu(A.data(), B.data(), output_gpu.data(), N);
    auto end_gpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_gpu.data(), 2 * N, "GPU Interleave");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count()
              << " ms" << std::endl << std::endl;

    bool results_match = true;
    for (int i = 0; i < 2 * N; i++) {
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
