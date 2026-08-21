#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/reverse.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/reverse.h"
#else
#include "cuda/reverse.h"
#endif
#include <iostream>
#include "benchmark.h"
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

    // Both reversals are in-place, so every iteration must start from the same
    // buffer contents -- otherwise an even iteration count would silently
    // un-reverse the array before the comparison below.
    auto restore = [N](float* buffer) {
        return [buffer, N] { for (int i = 0; i < N; i++) buffer[i] = (float)i; };
    };

    // CPU reversal (in-place)
    BenchResult cpu_bench = benchmarkWithReset([&] { reverse_array_cpu(input_cpu, N); },
                                               restore(input_cpu));
    #ifdef PRINT
    print(input_cpu, N, "CPU reversed");
    #endif
    printBench("CPU:", cpu_bench);

    // GPU reversal (in-place via host wrapper)
    BenchResult gpu_bench = benchmarkWithReset([&] { reverse_array_gpu(input_gpu, N); },
                                               restore(input_gpu));
    #ifdef PRINT
    print(input_gpu, N, "GPU reversed");
    #endif
    printBench("GPU:", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

    bool match = true;
    for (int i = 0; i < N; i++) {
        if (input_cpu[i] != input_gpu[i]) { match = false; break; }
    }
    std::cout << (match ? "Results match!" : "Results do NOT match!") << std::endl;

    free(input_cpu);
    free(input_gpu);
    return match ? 0 : 1;
}
