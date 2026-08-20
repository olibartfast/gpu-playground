#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/geglu.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/geglu.h"
#else
#include "cuda/geglu.h"
#endif
#include <iostream>
#include "benchmark.h"
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

    BenchResult cpu_bench = benchmark([&] {
        geglu_cpu(input.data(), output_cpu.data(), N/2);
    });
    printBench("CPU:", cpu_bench);

    BenchResult gpu_bench = benchmark([&] {
        geglu_gpu(input.data(), output_gpu.data(), N/2);
    });
    printBench("GPU:", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

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
