// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o clip clip.cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <vector>

#define PRINT 
#define BSIZE 256

void clip(const float* d_input, float* d_output, int N, float lo, float hi);

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

__global__ void clip_kernel(const float* input, float* output, int N, float lo, float hi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val = input[idx];
        // Clamp value to [lo, hi] range
        output[idx] = fminf(fmaxf(val, lo), hi);
    }
}

void clip_cpu(const float* input, float* output, int N, float lo, float hi) {
    for (int i = 0; i < N; i++) {
        float val = input[i];
        if (val < lo) {
            output[i] = lo;
        } else if (val > hi) {
            output[i] = hi;
        } else {
            output[i] = val;
        }
    }
}

void clip(const float* d_input, float* d_output, int N, float lo, float hi) {
    int threadsPerBlock = BSIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, lo, hi);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Use N=8 for printing and correctness check.
    // For performance, use a much larger N and comment out the #define PRINT.
    int N = 8;
    // int N = 1 << 20; // 1,048,576 elements for performance test

    float lo = 0.0f;
    float hi = 3.5f;

    std::vector<float> input(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_gpu(N);

    // Initialize array with some sample values
    if (N == 8) {
        float sample_values[] = {1.5f, -2.0f, 3.0f, 4.5f, -1.0f, 2.5f, 5.0f, 0.5f};
        for (int i = 0; i < N; i++) input[i] = sample_values[i];
    } else {
        // For large N, fill with random-like data
        for (int i = 0; i < N; i++) {
            input[i] = (float)(i % 100) - 50.0f;  // Range from -50 to 49
        }
    }
    
    #ifdef PRINT
    print(input.data(), N, "Input");
    std::cout << "Clipping range: [" << lo << ", " << hi << "]" << std::endl;
    #endif

    // --- CPU Clip ---
    auto start_cpu = std::chrono::steady_clock::now();
    clip_cpu(input.data(), output_cpu.data(), N, lo, hi);
    auto end_cpu = std::chrono::steady_clock::now();
    
    #ifdef PRINT
    print(output_cpu.data(), N, "CPU Clip");
    #endif

    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms" << std::endl << std::endl;
    
    // --- GPU Clip ---
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);

    auto start_gpu = std::chrono::steady_clock::now();
    
    cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    
    // Call clip kernel
    clip(d_input, d_output, N, lo, hi);
    
    // Ensure all GPU work is done before stopping the timer
    cudaDeviceSynchronize();

    cudaMemcpy(output_gpu.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    auto end_gpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_gpu.data(), N, "GPU Clip");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count() << " ms" << std::endl << std::endl;

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
    
    if (results_match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}