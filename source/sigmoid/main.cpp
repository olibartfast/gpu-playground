#include "sigmoid.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

int main()
{
    int N = 1024;
    std::vector<float> input(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_gpu(N);

    for(int i = 0; i < N; i++) {
        input[i] = (float)(i % 100) - 50.0f;
    }

    auto start_cpu = std::chrono::steady_clock::now();
    sigmoid_cpu(input.data(), output_cpu.data(), N);
    auto end_cpu = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU duration: " << duration_cpu.count() << " ms" << std::endl;

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);
    cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    auto start_gpu = std::chrono::steady_clock::now();
    sigmoid(d_input, d_output, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration_gpu = end_gpu - start_gpu;
    std::cout << "GPU duration: " << duration_gpu.count() << " ms" << std::endl;

    cudaMemcpy(output_gpu.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        if(fabs(output_cpu[i] - output_gpu[i]) > 1e-5) {
            std::cout << "[kernel1] Mismatch at index " << i << ": CPU " << output_cpu[i]
                      << " vs GPU " << output_gpu[i] << std::endl;
            cudaFree(d_input);
            cudaFree(d_output);
            return -1;
        }
    }
    std::cout << "[kernel1] Results match!" << std::endl;

    // Benchmark and validate sigmoid_kernel2 (vectorized float4)
    std::vector<float> output_gpu2(N);

    auto start_gpu2 = std::chrono::steady_clock::now();
    sigmoid2(d_input, d_output, N);
    cudaDeviceSynchronize();
    auto end_gpu2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration_gpu2 = end_gpu2 - start_gpu2;
    std::cout << "GPU2 (vectorized) duration: " << duration_gpu2.count() << " ms" << std::endl;

    cudaMemcpy(output_gpu2.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    for(int i = 0; i < N; i++) {
        if(fabs(output_cpu[i] - output_gpu2[i]) > 1e-5) {
            std::cout << "[kernel2] Mismatch at index " << i << ": CPU " << output_cpu[i]
                      << " vs GPU " << output_gpu2[i] << std::endl;
            return -1;
        }
    }
    std::cout << "[kernel2] Results match!" << std::endl;
    return 0;
}
