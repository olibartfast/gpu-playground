#include "cuda/fp16_dot_product.h"
#include "benchmark.h"
#include "benchmark_helpers.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float ABSOLUTE_TOLERANCE = 0.05f;
constexpr float RELATIVE_TOLERANCE = 0.05f;

struct TestCase {
    std::string name;
    std::vector<float> a;
    std::vector<float> b;
};

TestCase make_random_test(const std::string& name,
                          int n,
                          float low,
                          float high,
                          std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(low, high);
    TestCase test{name, std::vector<float>(n), std::vector<float>(n)};
    for (int i = 0; i < n; ++i) {
        test.a[i] = distribution(generator);
        test.b[i] = distribution(generator);
    }
    return test;
}

bool run_test(const TestCase& test,
             int cpu_warmup = kBenchWarmup,
             int cpu_iterations = kBenchIterations) {
    const int n = static_cast<int>(test.a.size());

    float expected = 0.0f;
    BenchResult cpu = benchmark(
        [&] { expected = fp16_dot_product_cpu(test.a.data(), test.b.data(), n); },
        cpu_warmup,
        cpu_iterations);

    float actual = 0.0f;
    DeviceBenchResult gpu = benchmarkDevice([&] {
        float kernel_ms = 0.0f;
        actual = fp16_dot_product_gpu(test.a.data(), test.b.data(), n, &kernel_ms);
        return kernel_ms;
    });

    const double operations = 2.0 * static_cast<double>(n);
    const double cpu_gflops =
        gpu_benchmark::giga_operations_per_second(operations, cpu.median_ms);
    const double gpu_kernel_gflops =
        gpu_benchmark::giga_operations_per_second(operations, gpu.device.median_ms);
    const double gpu_end_to_end_gflops =
        gpu_benchmark::giga_operations_per_second(operations, gpu.end_to_end.median_ms);

    const float error = std::fabs(expected - actual);
    const float tolerance =
        ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * std::fabs(expected);
    const bool passed = error <= tolerance;

    std::cout << std::left << std::setw(24) << test.name
              << (passed ? "PASS" : "FAIL")
              << "  N=" << n
              << "  expected=" << expected
              << "  actual=" << actual
              << "  error=" << error
              << '\n';
    printBench("CPU:", cpu);
    printBench("GPU kernel:", gpu.device);
    printBench("GPU end-to-end:", gpu.end_to_end);
    std::cout << "  CPU:            " << cpu_gflops << " GFLOP/s\n"
              << "  GPU kernel:     " << gpu_kernel_gflops << " GFLOP/s\n"
              << "  GPU end-to-end: " << gpu_end_to_end_gflops << " GFLOP/s\n";
    printSpeedup("Speedup kernel:", cpu, gpu.device);
    printSpeedup("Speedup end-to-end:", cpu, gpu.end_to_end);
    return passed;
}

std::vector<TestCase> make_functional_tests() {
    // LeetGPU's random functional cases do not specify a seed. Keep this local
    // reproduction deterministic while preserving their sizes and ranges.
    std::mt19937 generator(0);
    std::vector<TestCase> tests;
    tests.push_back({"basic_small",
                     {1.0f, 2.0f, 3.0f, 4.0f},
                     {5.0f, 6.0f, 7.0f, 8.0f}});
    tests.push_back({"all_zeros",
                     std::vector<float>(16, 0.0f),
                     std::vector<float>(16, 0.0f)});
    tests.push_back({"negative_numbers",
                     {-1.0f, -2.0f, -3.0f, -4.0f},
                     {-5.0f, -6.0f, -7.0f, -8.0f}});
    tests.push_back({"mixed_positive_negative",
                     {1.0f, -2.0f, 3.0f, -4.0f},
                     {-1.0f, 2.0f, -3.0f, 4.0f}});
    tests.push_back({"orthogonal_vectors",
                     {1.0f, 0.0f, 0.0f},
                     {0.0f, 1.0f, 0.0f}});
    tests.push_back(
        make_random_test("medium_sized_vector", 1000, -1.0f, 1.0f, generator));
    tests.push_back(
        make_random_test("large_vector", 10000, -0.1f, 0.1f, generator));
    return tests;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(6);

    if (argc == 2 && std::string(argv[1]) == "--performance") {
        std::mt19937 generator(0);
        const TestCase performance =
            make_random_test("performance", 100000000, -1.0f, 1.0f, generator);
        // n=1e8: a slow CPU reference would otherwise dominate harness runtime.
        return run_test(performance, 1, 3) ? 0 : 1;
    }

    bool passed = true;
    for (const TestCase& test : make_functional_tests()) {
        passed = run_test(test) && passed;
    }

    std::cout << (passed ? "All LeetGPU functional tests passed."
                         : "One or more LeetGPU functional tests failed.")
              << std::endl;
    return passed ? 0 : 1;
}
