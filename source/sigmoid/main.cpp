#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/sigmoid.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/sigmoid.h"
#else
#include "cuda/sigmoid.h"
#endif
#include <iostream>
#include "benchmark.h"
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

    BenchResult cpu_bench = benchmark([&] {
        sigmoid_cpu(input.data(), output_cpu.data(), N);
    });
    printBench("CPU:", cpu_bench);

    BenchResult gpu_bench = benchmark([&] {
        sigmoid_gpu(input.data(), output_gpu.data(), N);
    });
    printBench("GPU (scalar):", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

    for(int i = 0; i < N; i++) {
        if(fabs(output_cpu[i] - output_gpu[i]) > 1e-5) {
            std::cout << "[kernel1] Mismatch at index " << i << ": CPU " << output_cpu[i]
                      << " vs GPU " << output_gpu[i] << std::endl;
            return -1;
        }
    }
    std::cout << "[kernel1] Results match!" << std::endl;

    std::vector<float> output_gpu2(N);
    BenchResult gpu2_bench = benchmark([&] {
        sigmoid2_gpu(input.data(), output_gpu2.data(), N);
    });
    printBench("GPU (vectorized):", gpu2_bench);
    printSpeedup("Speedup (CPU/GPU2):", cpu_bench, gpu2_bench);
    printSpeedup("Vectorized vs scalar:", gpu_bench, gpu2_bench);

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
