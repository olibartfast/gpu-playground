#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/gaussian_blur.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/gaussian_blur.h"
#else
#include "cuda/gaussian_blur.h"
#endif
#include "benchmark.h"
#include "benchmark_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

// LeetGPU challenge 28 compares with torch.allclose(atol=1e-5, rtol=1e-5).
constexpr float ABSOLUTE_TOLERANCE = 1.0e-5f;
constexpr float RELATIVE_TOLERANCE = 1.0e-5f;

struct TestCase {
    std::string name;
    int input_rows;
    int input_cols;
    int kernel_rows;
    int kernel_cols;
    std::vector<float> input;
    std::vector<float> kernel;
};

TestCase make_random_test(const std::string& name, int input_rows,
                          int input_cols, int kernel_rows, int kernel_cols,
                          float input_low, float input_high, float kernel_low,
                          float kernel_high, std::mt19937& generator) {
    std::uniform_real_distribution<float> input_distribution(input_low,
                                                              input_high);
    std::uniform_real_distribution<float> kernel_distribution(kernel_low,
                                                               kernel_high);
    TestCase test{name,
                  input_rows,
                  input_cols,
                  kernel_rows,
                  kernel_cols,
                  std::vector<float>(static_cast<std::size_t>(input_rows) *
                                     input_cols),
                  std::vector<float>(static_cast<std::size_t>(kernel_rows) *
                                     kernel_cols)};
    for (float& value : test.input) {
        value = input_distribution(generator);
    }
    for (float& value : test.kernel) {
        value = kernel_distribution(generator);
    }
    return test;
}

std::vector<TestCase> make_functional_tests() {
    std::mt19937 generator(0);
    std::vector<TestCase> tests;
    tests.push_back({"basic_example",
                     5,
                     5,
                     3,
                     3,
                     {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                      8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                      15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f,
                      22.0f, 23.0f, 24.0f, 25.0f},
                     {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f,
                      0.125f, 0.0625f}});
    tests.push_back({"identity_kernel",
                     3,
                     3,
                     3,
                     3,
                     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
                     {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
    tests.push_back({"all_ones_input", 4, 4, 3, 3,
                     std::vector<float>(16, 1.0f),
                     std::vector<float>(9, 0.111111f)});
    tests.push_back({"single_pixel", 1, 1, 1, 1, {42.0f}, {1.0f}});
    tests.push_back(make_random_test("large_random", 32, 32, 5, 5, -10.0f, 10.0f,
                                     0.0f, 1.0f, generator));
    return tests;
}

bool compare(const std::vector<float>& expected,
            const std::vector<float>& actual, float& max_error) {
    max_error = 0.0f;
    bool passed = true;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float error = std::fabs(expected[i] - actual[i]);
        max_error = std::max(max_error, error);
        if (error > ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * std::fabs(expected[i])) {
            passed = false;
        }
    }
    return passed;
}

bool run_functional_test(const TestCase& test) {
    const std::size_t pixels =
        static_cast<std::size_t>(test.input_rows) * test.input_cols;
    std::vector<float> expected(pixels);
    std::vector<float> actual(pixels);

    BenchResult cpu_bench = benchmark([&] {
        gaussian_blur_cpu(test.input.data(), test.kernel.data(), expected.data(),
                          test.input_rows, test.input_cols, test.kernel_rows,
                          test.kernel_cols);
    });
    BenchResult gpu_bench = benchmark([&] {
        gaussian_blur_gpu(test.input.data(), test.kernel.data(), actual.data(),
                          test.input_rows, test.input_cols, test.kernel_rows,
                          test.kernel_cols);
    });

    float max_error = 0.0f;
    const bool passed = compare(expected, actual, max_error);
    std::cout << std::left << std::setw(20) << test.name
              << (passed ? "PASS" : "FAIL") << "  " << test.input_rows << "x"
              << test.input_cols << " * " << test.kernel_rows << "x"
              << test.kernel_cols << "  max_error=" << max_error << '\n';
    printBench("CPU:", cpu_bench);
    printBench("GPU:", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);
    return passed;
}

bool run_performance_test(const TestCase& test) {
    const std::size_t pixels =
        static_cast<std::size_t>(test.input_rows) * test.input_cols;
    std::vector<float> expected(pixels);
    std::vector<float> actual(pixels);

    BenchResult cpu_bench = benchmark([&] {
        gaussian_blur_cpu(test.input.data(), test.kernel.data(), expected.data(),
                          test.input_rows, test.input_cols, test.kernel_rows,
                          test.kernel_cols);
    });

    std::vector<float> kernel_times;
    BenchResult gpu_bench = benchmark([&] {
        float kernel_ms = 0.0f;
        gaussian_blur_gpu(test.input.data(), test.kernel.data(), actual.data(),
                          test.input_rows, test.input_cols, test.kernel_rows,
                          test.kernel_cols, &kernel_ms);
        kernel_times.push_back(kernel_ms);
    });
    // Drop warm-up samples, then take the median of the timed iterations.
    kernel_times.erase(kernel_times.begin(),
                       kernel_times.begin() + gpu_bench.warmup);
    std::sort(kernel_times.begin(), kernel_times.end());
    const double kernel_ms = kernel_times[kernel_times.size() / 2];

    float max_error = 0.0f;
    const bool passed = compare(expected, actual, max_error);

    // 2 FLOPs per tap; naive traffic model: each output reads its full window.
    const double taps = static_cast<double>(pixels) * test.kernel_rows *
                        test.kernel_cols;
    const double min_bytes = 2.0 * pixels * sizeof(float);

    std::cout << test.name << "  " << (passed ? "PASS" : "FAIL") << "  "
              << test.input_rows << "x" << test.input_cols << " * "
              << test.kernel_rows << "x" << test.kernel_cols
              << "  max_error=" << max_error << '\n';
    printBench("CPU:", cpu_bench);
    printBench("GPU end-to-end:", gpu_bench);
    std::cout << "GPU kernel (median): " << kernel_ms << " ms, "
              << gpu_benchmark::giga_operations_per_second(2.0 * taps, kernel_ms)
              << " GFLOP/s, "
              << gpu_benchmark::gigabytes_per_second(min_bytes, kernel_ms)
              << " GB/s (compulsory traffic)\n";
    printSpeedup("Speedup (CPU/GPU end-to-end):", cpu_bench, gpu_bench);
    std::cout << "Speedup (CPU/GPU kernel): "
              << gpu_benchmark::speedup(cpu_bench.median_ms, kernel_ms) << "x\n";
    return passed;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(6);
    if (argc == 2 && std::string(argv[1]) == "--performance") {
        std::mt19937 generator(0);
        const TestCase performance = make_random_test(
            "performance", 512, 512, 7, 7, 0.0f, 255.0f, 0.0001f, 0.02f, generator);
        const bool passed = run_performance_test(performance);
        std::cout << "Overall result: " << (passed ? "PASSED" : "FAILED") << std::endl;
        return passed ? 0 : 1;
    }

    bool passed = true;
    for (const TestCase& test : make_functional_tests()) {
        passed = run_functional_test(test) && passed;
    }
    std::cout << "Overall result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}
