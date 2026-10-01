#include "benchmark.h"
#include "benchmark_helpers.h"
#include "cuda/categorical_cross_entropy.h"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float ABSOLUTE_TOLERANCE = 1.0e-5f;
constexpr float RELATIVE_TOLERANCE = 1.0e-5f;

struct TestCase {
  std::string name;
  int n;
  int c;
  std::vector<float> logits;
  std::vector<int> labels;
};

TestCase make_random_test(const std::string &name, int n, int c, float low,
                          float high, std::mt19937 &generator) {
  std::uniform_real_distribution<float> logits_distribution(low, high);
  std::uniform_int_distribution<int> labels_distribution(0, c - 1);
  TestCase test{name, n, c, std::vector<float>(static_cast<std::size_t>(n) * c),
                std::vector<int>(n)};
  for (float &value : test.logits) {
    value = logits_distribution(generator);
  }
  for (int &label : test.labels) {
    label = labels_distribution(generator);
  }
  return test;
}

bool run_test(const TestCase &test) {
  float expected = 0.0f;
  BenchResult cpu = benchmark([&] {
    expected = categorical_cross_entropy_cpu(
        test.logits.data(), test.labels.data(), test.n, test.c);
  });

  float actual = 0.0f;
  DeviceBenchResult gpu = benchmarkDevice([&] {
    float kernel_ms = 0.0f;
    actual = categorical_cross_entropy_gpu(
        test.logits.data(), test.labels.data(), test.n, test.c, &kernel_ms);
    return kernel_ms;
  });

  const double class_items = static_cast<double>(test.n) * test.c;
  const double bytes = class_items * sizeof(float) +
                       static_cast<double>(test.n) * sizeof(int) +
                       sizeof(float);
  const float error = std::fabs(expected - actual);
  const float tolerance =
      ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * std::fabs(expected);
  const bool passed = error <= tolerance;

  std::cout << std::left << std::setw(28) << test.name
            << (passed ? "PASS" : "FAIL") << "  N=" << test.n
            << "  C=" << test.c << "  expected=" << expected
            << "  actual=" << actual << "  error=" << error << '\n';
  printBench("CPU:", cpu);
  printBench("GPU kernel:", gpu.device);
  printBench("GPU end-to-end:", gpu.end_to_end);
  std::cout << "  CPU:            "
            << gpu_benchmark::million_items_per_second(class_items,
                                                       cpu.median_ms)
            << " Mclass/s\n"
            << "  GPU kernel:     "
            << gpu_benchmark::million_items_per_second(class_items,
                                                       gpu.device.median_ms)
            << " Mclass/s, "
            << gpu_benchmark::gigabytes_per_second(bytes, gpu.device.median_ms)
            << " GB/s\n"
            << "  GPU end-to-end: "
            << gpu_benchmark::million_items_per_second(
                   class_items, gpu.end_to_end.median_ms)
            << " Mclass/s\n";
  printSpeedup("Speedup kernel:", cpu, gpu.device);
  printSpeedup("Speedup end-to-end:", cpu, gpu.end_to_end);
  return passed;
}

std::vector<TestCase> make_functional_tests() {
  std::mt19937 generator(0);
  std::vector<TestCase> tests;
  tests.push_back(
      {"basic_example", 2, 3, {1.0f, 2.0f, 0.5f, 0.1f, 3.0f, 1.5f}, {1, 1}});
  tests.push_back({"example_2",
                   3,
                   4,
                   {-0.5f, 1.5f, 0.0f, 1.0f, 2.0f, -1.0f, 0.5f, 0.5f, 0.0f,
                    0.0f, 0.0f, 0.0f},
                   {3, 0, 1}});
  tests.push_back({"single_sample", 1, 5, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f}, {4}});
  tests.push_back({"uniform_logits_correct_label",
                   2,
                   5,
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                   {0, 0}});
  tests.push_back({"high_confidence_correct",
                   2,
                   4,
                   {-5.0f, -5.0f, 10.0f, -5.0f, 10.0f, -5.0f, -5.0f, -5.0f},
                   {2, 0}});
  tests.push_back({"high_confidence_incorrect",
                   2,
                   3,
                   {10.0f, -5.0f, -5.0f, -5.0f, 10.0f, -5.0f},
                   {1, 2}});
  tests.push_back(
      make_random_test("larger_batch_random", 100, 5, -5.0f, 5.0f, generator));
  return tests;
}

} // namespace

int main(int argc, char **argv) {
  std::cout << std::fixed << std::setprecision(6);
  if (argc == 2 && std::string(argv[1]) == "--performance") {
    std::mt19937 generator(0);
    const TestCase performance =
        make_random_test("performance", 10000, 1000, -10.0f, 10.0f, generator);
    return run_test(performance) ? 0 : 1;
  }

  bool passed = true;
  for (const TestCase &test : make_functional_tests()) {
    passed = run_test(test) && passed;
  }
  std::cout << (passed ? "All LeetGPU functional tests passed."
                       : "One or more LeetGPU functional tests failed.")
            << std::endl;
  return passed ? 0 : 1;
}
