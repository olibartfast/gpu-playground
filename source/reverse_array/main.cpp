#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/reverse.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/reverse.h"
#else
#include "cuda/reverse.h"
#endif
#include <iostream>
#include <chrono>
#include <cstring>

#define PRINT

void print(const float* input, int N, const std::string& message = "") {
    if (!message.empty()) std::cout << message << ": ";
    for (int i = 0; i < N; i++) std::cout << input[i] << " ";
    std::cout << std::endl;
}

int main() {
    int N = 10;
    float* input_cpu = (float*)malloc(sizeof(float) * N);
    float* input_gpu = (float*)malloc(sizeof(float) * N);
    if (!input_cpu || !input_gpu) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    for (int i = 0; i < N; i++) input_cpu[i] = input_gpu[i] = (float)i;

    #ifdef PRINT
    print(input_cpu, N, "Starting list");
    #endif

    // CPU reversal (in-place)
    auto start = std::chrono::steady_clock::now();
    reverse_array_cpu(input_cpu, N);
    auto end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print(input_cpu, N, "CPU reversed");
    #endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // GPU reversal (in-place via host wrapper)
    start = std::chrono::steady_clock::now();
    reverse_array_gpu(input_gpu, N);
    end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print(input_gpu, N, "GPU reversed");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    bool match = true;
    for (int i = 0; i < N; i++) {
        if (input_cpu[i] != input_gpu[i]) { match = false; break; }
    }
    std::cout << (match ? "Results match!" : "Results do NOT match!") << std::endl;

    free(input_cpu);
    free(input_gpu);
    return match ? 0 : 1;
}
