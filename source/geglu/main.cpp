#include "geglu.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

int main(int argc, char const *argv[])
{
    int N = 1024;
    std::vector<float> input(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_gpu(N);
    for(int i=0; i<N; i++) {
        input[i] = (float)(i % 100);
    }

    auto start_cpu = std::chrono::steady_clock::now();
    geglu_cpu(input.data(), output_cpu.data(), N/2);
    auto end_cpu = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU duration: " << duration_cpu.count() << " ms" << std::endl;

    // GPU computation
    auto start_gpu = std::chrono::steady_clock::now();
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);
    cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    geglu(d_input, d_output, N/2);
    cudaMemcpy(output_gpu.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    auto end_gpu = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration_gpu = end_gpu - start_gpu;
    std::cout << "GPU duration: " << duration_gpu.count() << " ms" << std::endl;
    cudaFree(d_input);
    cudaFree(d_output);
    for(int i=0; i<N; i++) {
        if(fabs(output_cpu[i] - output_gpu[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": CPU " << output_cpu[i] << " vs GPU " << output_gpu[i] << std::endl;
            return -1;
        }
    }
    std::cout << "Results match!" << std::endl;
    return 0;
}
