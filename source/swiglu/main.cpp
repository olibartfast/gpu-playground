#include "swiglu.h"
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
    // Use N=8 for printing and correctness check.
    // For performance, use a much larger N and comment out the #define PRINT.
    int N = 8;
    // int N = 1 << 20; // 1,048,576 elements for performance test

    std::vector<float> input(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_gpu(N);

    // Initialize array with some sample values
    if (N == 8) {
        float sample_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
        for (int i = 0; i < N; i++) input[i] = sample_values[i];
    } else {
        // For large N, fill with random-like data
        for (int i = 0; i < N; i++) input[i] = (float)(i % 100);
    }

    #ifdef PRINT
    print(input.data(), N, "Input");
    #endif

    // --- CPU SwiGLU ---
    auto start_cpu = std::chrono::steady_clock::now();
    swiglu_cpu(input.data(), output_cpu.data(), N);
    auto end_cpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_cpu.data(), N/2, "CPU SwiGLU");  // Only print first half as SwiGLU reduces dimension
    #endif

    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms" << std::endl << std::endl;

    // --- GPU SwiGLU ---
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N/2);  // SwiGLU output is half the size

    auto start_gpu = std::chrono::steady_clock::now();

    cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    swiglu(d_input, d_output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(output_gpu.data(), d_output, sizeof(float) * N/2, cudaMemcpyDeviceToHost);

    auto end_gpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_gpu.data(), N/2, "GPU SwiGLU");  // Only print first half as SwiGLU reduces dimension
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count() << " ms" << std::endl << std::endl;

    // --- Verification ---
    bool results_match = true;
    const float tolerance = 1e-5f;
    for (int i = 0; i < N/2; i++) {  // Only compare first half as SwiGLU reduces dimension
        if (fabsf(output_cpu[i] - output_gpu[i]) > tolerance) {
            results_match = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << output_cpu[i]
                      << " GPU=" << output_gpu[i] << std::endl;
            break;
        }
    }

    if (results_match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
