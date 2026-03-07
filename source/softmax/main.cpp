#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/softmax.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/softmax.h"
#else
#include "cuda/softmax.h"
#endif
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>

#define PRINT // Comment this out for large N to avoid printing thousands of numbers

void print(const float* input, int N, const std::string& message = "") {
    if (!message.empty()) std::cout << message << ": ";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < N; i++) std::cout << input[i] << " ";
    std::cout << std::endl;
}

int main() {
    int N = 8;
    // int N = 1 << 20; // for performance testing; comment out #define PRINT above

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

    // --- CPU softmax ---
    auto start_cpu = std::chrono::steady_clock::now();
    softmax_cpu(input.data(), output_cpu.data(), N);
    auto end_cpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_cpu.data(), N, "CPU softmax");
    #endif
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) cpu_sum += output_cpu[i];
    std::cout << "CPU softmax sum: " << std::fixed << std::setprecision(6) << cpu_sum << std::endl;
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms\n\n";

    // --- GPU softmax ---
    auto start_gpu = std::chrono::steady_clock::now();
    softmax_gpu(input.data(), output_gpu.data(), N);
    auto end_gpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_gpu.data(), N, "GPU softmax");
    #endif
    float gpu_sum = 0.0f;
    for (int i = 0; i < N; i++) gpu_sum += output_gpu[i];
    std::cout << "GPU softmax sum: " << std::fixed << std::setprecision(6) << gpu_sum << std::endl;
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count() << " ms\n\n";

    // --- Verification ---
    bool results_match = true;
    const float tolerance = 1e-5f;
    for (int i = 0; i < N; i++) {
        if (fabsf(output_cpu[i] - output_gpu[i]) > tolerance) {
            results_match = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << output_cpu[i]
                      << " GPU=" << output_gpu[i] << std::endl;
            break;
        }
    }
    std::cout << (results_match ? "Results match!" : "Results do not match!") << std::endl;
    return results_match ? 0 : 1;
}
