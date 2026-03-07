#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/rgb_to_grayscale.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/rgb_to_grayscale.h"
#else
#include "cuda/rgb_to_grayscale.h"
#endif
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// #define PRINT // use only to debug for very small arrays

void print(const float* data, int N, int channels, const std::string& message = "") {
    if (!message.empty()) std::cout << message << ": ";
    for (int i = 0; i < N * channels; i++) std::cout << data[i] << " ";
    std::cout << std::endl;
}

void print(const float* data, int N, const std::string& message = "") {
    print(data, N, 1, message);
}

int main() {
    int width = 1920;
    int height = 1080;
    int total_pixels = width * height;
    int input_size = total_pixels * 3;

    float* input = (float*)malloc(sizeof(float) * input_size);
    float* output_cpu = (float*)malloc(sizeof(float) * total_pixels);
    float* output_gpu = (float*)malloc(sizeof(float) * total_pixels);
    if (!input || !output_cpu || !output_gpu) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    for (int i = 0; i < total_pixels; i++) {
        input[i * 3 + 0] = static_cast<float>(i * 30 % 256);
        input[i * 3 + 1] = static_cast<float>((i * 50 + 10) % 256);
        input[i * 3 + 2] = static_cast<float>((i * 70 + 20) % 256);
    }

    #ifdef PRINT
    print(input, total_pixels, 3, "Input RGB");
    #endif

    auto start = std::chrono::steady_clock::now();
    rgb_to_grayscale_cpu(input, output_cpu, total_pixels);
    auto end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print(output_cpu, total_pixels, "CPU grayscale");
    #endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();
    rgb_to_grayscale_gpu(input, output_gpu, total_pixels);
    end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print(output_gpu, total_pixels, "GPU grayscale");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    float max_diff = 0.0f;
    for (int i = 0; i < total_pixels; i++) {
        float diff = std::abs(output_cpu[i] - output_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max CPU vs GPU difference: " << max_diff << std::endl;
    if (max_diff < 1e-5f) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }

    free(input);
    free(output_cpu);
    free(output_gpu);
    return max_diff < 1e-5f ? 0 : 1;
}
