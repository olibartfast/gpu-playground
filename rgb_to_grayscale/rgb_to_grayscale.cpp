// nvcc -x cu -o rgb2gray rgb2gray.cpp -std=c++17
// on Tesla T4 GPU
// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o rgb2gray rgb2gray.cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

// #define PRINT // use only to debug for very small arrays

void print(float* data, int N, int channels, const std::string& message = "") {
    if (!message.empty()) {
        std::cout << message << ": ";
    }
    for (int i = 0; i < N * channels; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void print(float* data, int N, const std::string& message = "") {
    print(data, N, 1, message);
}

__global__ void rgb_to_grayscale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_pixels; i += stride) {
        int input_idx = i * 3;
        float r = input[input_idx];
        float g = input[input_idx + 1];
        float b = input[input_idx + 2];
        output[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void rgb_to_grayscale_cpu(const float* input, float* output, int total_pixels) {
    for (int i = 0; i < total_pixels; i++) {
        int idx = i * 3;
        output[i] = 0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
    }
}

int main() {
    int width = 1920;
    int height = 1080;
    int total_pixels = width * height;
    int input_size = total_pixels * 3;

    float* input = (float*)malloc(sizeof(float) * input_size);
    float* output = (float*)malloc(sizeof(float) * total_pixels);
    if (!input || !output) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    // Initialize with some RGB values
    for (int i = 0; i < total_pixels; i++) {
        input[i * 3 + 0] = static_cast<float>(i * 30 % 256);       // R
        input[i * 3 + 1] = static_cast<float>((i * 50 + 10) % 256); // G
        input[i * 3 + 2] = static_cast<float>((i * 70 + 20) % 256); // B
    }

    #ifdef PRINT
    print(input, total_pixels, 3, "Input RGB");
    #endif

    // Validate CPU vs GPU
    float* cpu_output = (float*)malloc(sizeof(float) * total_pixels);
    rgb_to_grayscale_cpu(input, cpu_output, total_pixels);

    // CPU grayscale
    auto start = std::chrono::steady_clock::now();
    rgb_to_grayscale_cpu(input, output, total_pixels);
    auto end = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output, total_pixels, "CPU grayscale");
    #endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // GPU grayscale
    float* d_input;
    float* d_output;
    cudaError_t err = cudaMalloc(&d_input, sizeof(float) * input_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc input failed: " << cudaGetErrorString(err) << std::endl;
        free(input); free(output);
        return 1;
    }
    err = cudaMalloc(&d_output, sizeof(float) * total_pixels);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc output failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input); free(input); free(output);
        return 1;
    }

    start = std::chrono::steady_clock::now();
    err = cudaMemcpy(d_input, input, sizeof(float) * input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input); cudaFree(d_output); free(input); free(output);
        return 1;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = std::min(65535, (total_pixels + threadsPerBlock - 1) / threadsPerBlock);
    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, total_pixels);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input); cudaFree(d_output); free(input); free(output);
        return 1;
    }

    err = cudaMemcpy(output, d_output, sizeof(float) * total_pixels, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input); cudaFree(d_output); free(input); free(output);
        return 1;
    }

    end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print(output, total_pixels, "GPU grayscale");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;


    float max_diff = 0.0f;
    for (int i = 0; i < total_pixels; i++) {
        float diff = std::abs(cpu_output[i] - output[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max CPU vs GPU difference: " << max_diff << std::endl;
    if (max_diff < 1e-5f) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }

    free(cpu_output);

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    return 0;
}