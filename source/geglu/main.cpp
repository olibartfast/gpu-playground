#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/geglu.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/geglu.h"
#else
#include "cuda/geglu.h"
#endif
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

    auto start_gpu = std::chrono::steady_clock::now();
    geglu_gpu(input.data(), output_gpu.data(), N/2);
    auto end_gpu = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration_gpu = end_gpu - start_gpu;
    std::cout << "GPU duration: " << duration_gpu.count() << " ms" << std::endl;

    for(int i=0; i<N/2; i++) {
        if(fabs(output_cpu[i] - output_gpu[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": CPU " << output_cpu[i]
                      << " vs GPU " << output_gpu[i] << std::endl;
            return -1;
        }
    }
    std::cout << "Results match!" << std::endl;
    return 0;
}
