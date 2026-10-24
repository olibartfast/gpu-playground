#if defined(GPU_OPENGL_BACKEND) || defined(GPU_VULKAN_BACKEND)
#include "benchmark.h"
#include "glsl/sigmoid.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <vector>

int main() {
  try {
    for (int n : {0, 1, 257, 1024, 1000003}) {
      std::vector<float> input(n), reference(n), output(n);
      for (int i = 0; i < n; ++i)
        input[i] = float((i + 500) % 1001 - 500) * 0.1f;
      std::cout << "N=" << n << " FP32 ATOL=1e-5\n";
      printBench("CPU:", benchmark([&] {
                   for (int i = 0; i < n; ++i)
                     reference[i] = 1.0f / (1.0f + std::exp(-input[i]));
                 }));
      for (auto fn : {sigmoid_gpu, sigmoid2_gpu}) {
        std::cout << (fn == sigmoid_gpu ? "Scalar\n"
                                        : "Four elements per invocation\n");
        auto validate = [&] {
          for (int i = 0; i < n; ++i)
            if (!std::isfinite(output[i]) ||
                std::abs(output[i] - reference[i]) > 1e-5f)
              throw std::runtime_error("Sigmoid mismatch at index " +
                                       std::to_string(i));
        };
        std::fill(output.begin(), output.end(), NAN);
        float initial_ms = 0;
        fn(input.data(), output.data(), n, &initial_ms);
        validate();
        auto result = benchmarkDeviceWithReset(
            [&] {
              float ms = 0;
              fn(input.data(), output.data(), n, &ms);
              return ms;
            },
            [&] { std::fill(output.begin(), output.end(), NAN); });
        validate();
        printBench("GPU end-to-end:", result.end_to_end);
        printBench("GPU kernel:", result.device);
      }
    }
    std::cout << "Overall result: PASSED\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Overall result: FAILED: " << error.what() << '\n';
    return 1;
  }
}
#else
#if defined(GPU_HIP_BACKEND)
#include "hip/sigmoid.h"
#elif defined(GPU_OPENCL_CPP_BACKEND)
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

#endif
