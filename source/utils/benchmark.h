#pragma once
// Timing helpers shared by every kernel harness.
//
// A single timed call says very little about the kernel: the first invocation
// pays one-off costs (CUDA context creation, OpenCL program build, first-touch
// page faults, clock ramp-up) that can dwarf the work being measured. Harnesses
// therefore run a few untimed warm-up calls, then report statistics over
// several timed iterations so run-to-run noise is visible instead of hidden.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

// Default protocol: discard kBenchWarmup untimed calls, then time kBenchIterations.
constexpr int kBenchWarmup = 3;
constexpr int kBenchIterations = 10;

struct BenchResult {
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    int warmup = 0;
    int iterations = 0;
};

// `reset` runs before every call (warm-up and timed alike) and is never timed.
// Use it for kernels whose output buffer is also an input, e.g. GEMM's beta * C
// term, so each iteration measures the same work on the same starting state.
template <typename Fn, typename Reset>
BenchResult benchmarkWithReset(Fn&& fn, Reset&& reset,
                               int warmup = kBenchWarmup,
                               int iterations = kBenchIterations) {
    BenchResult result;
    if (iterations < 1) iterations = 1;
    if (warmup < 0) warmup = 0;
    result.warmup = warmup;
    result.iterations = iterations;

    for (int i = 0; i < warmup; i++) {
        reset();
        fn();
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; i++) {
        reset();
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(samples.begin(), samples.end());
    result.min_ms = samples.front();
    result.max_ms = samples.back();
    result.median_ms = (iterations % 2 == 1)
                           ? samples[iterations / 2]
                           : 0.5 * (samples[iterations / 2 - 1] + samples[iterations / 2]);

    double sum = 0.0;
    for (double s : samples) sum += s;
    result.mean_ms = sum / iterations;

    double sq = 0.0;
    for (double s : samples) sq += (s - result.mean_ms) * (s - result.mean_ms);
    result.stddev_ms = (iterations > 1) ? std::sqrt(sq / (iterations - 1)) : 0.0;

    return result;
}

template <typename Fn>
BenchResult benchmark(Fn&& fn,
                      int warmup = kBenchWarmup,
                      int iterations = kBenchIterations) {
    return benchmarkWithReset(fn, [] {}, warmup, iterations);
}

inline void printBench(const char* label, const BenchResult& r) {
    printf("%-22s median %9.4f ms | mean %9.4f ms +/- %.4f | min %9.4f | max %9.4f  (%d warm-up + %d timed)\n",
           label, r.median_ms, r.mean_ms, r.stddev_ms, r.min_ms, r.max_ms,
           r.warmup, r.iterations);
}

// Compare on medians: the mean is the metric an outlier iteration can move most.
inline void printSpeedup(const char* label, const BenchResult& baseline, const BenchResult& candidate) {
    if (candidate.median_ms > 0.0)
        printf("%-22s %.2fx (median baseline / median candidate)\n", label,
               baseline.median_ms / candidate.median_ms);
}
