#include "prefix_sum.h"
#include <iostream>
#include <chrono>
#include <cmath>

#define PRINT

// Function to print the array
void print(float* input, int N, const std::string& message = "") {
    if (!message.empty()) {
        std::cout << message << ": ";
    }
    for (int i = 0; i < N; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    int N = 4; // Test case size
    float* input = (float*)malloc(sizeof(float) * N);
    float* output = (float*)malloc(sizeof(float) * N);
    if (!input || !output) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    // Initialize array
    float test_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < N; i++) {
        input[i] = test_input[i];
        output[i] = 0.0f;
    }
#ifdef PRINT
    print(input, N, "Starting list");
#endif

    // CPU prefix scan
    auto start = std::chrono::steady_clock::now();
    prefix_scan_cpu(input, output, N);
    auto end = std::chrono::steady_clock::now();

#ifdef PRINT
    print(output, N, "CPU prefix scan");
#endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // GPU prefix scan
    float *d_input, *d_output, *d_block_sums = nullptr;
    cudaError_t err = cudaMalloc(&d_input, sizeof(float) * N);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed (input): " << cudaGetErrorString(err) << std::endl;
        free(input);
        free(output);
        return 1;
    }
    err = cudaMalloc(&d_output, sizeof(float) * N);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed (output): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        free(input);
        free(output);
        return 1;
    }

    start = std::chrono::steady_clock::now();
    err = cudaMemcpy(d_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        free(input);
        free(output);
        return 1;
    }

    int threadsPerBlock = 256;
    int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (N <= threadsPerBlock) {
        // Single-block case
        block_inclusive_scan<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, nullptr, N, threadsPerBlock);
    } else {
        // Multi-block case
        err = cudaMalloc(&d_block_sums, numberOfBlocks * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed (block_sums): " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_input);
            cudaFree(d_output);
            free(input);
            free(output);
            return 1;
        }
        block_inclusive_scan<<<numberOfBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, d_block_sums, N, threadsPerBlock);
        scan_block_sums<<<1, numberOfBlocks, numberOfBlocks * sizeof(float)>>>(d_block_sums, numberOfBlocks);
        add_block_sums<<<numberOfBlocks, threadsPerBlock>>>(d_output, d_block_sums, N, threadsPerBlock);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        if (d_block_sums) cudaFree(d_block_sums);
        free(input);
        free(output);
        return 1;
    }

    err = cudaMemcpy(output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        if (d_block_sums) cudaFree(d_block_sums);
        free(input);
        free(output);
        return 1;
    }

    end = std::chrono::steady_clock::now();
#ifdef PRINT
    print(output, N, "GPU prefix scan");
#endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // Verify output
    float expected[] = {1.0f, 3.0f, 6.0f, 10.0f};
    bool passed = true;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabs(output[i] - expected[i]);
        if (diff > 1e-5) {
            passed = false;
            max_diff = fmax(max_diff, diff);
        }
    }
    if (passed) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed! Max difference: " << max_diff << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    if (d_block_sums) cudaFree(d_block_sums);
    free(input);
    free(output);
    return 0;
}
