#pragma once

#include <chrono>
#include <utility>

namespace gpu_benchmark {

// Measures a callable after optional warmup runs and returns average wall-clock
// latency. GPU callables must synchronize before returning; the repository's
// host-pointer wrappers normally do this when copying results back to the host.
template <typename Callable>
double average_milliseconds(Callable&& callable,
                            int warmup_iterations = 1,
                            int measured_iterations = 10) {
    for (int i = 0; i < warmup_iterations; ++i) {
        callable();
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < measured_iterations; ++i) {
        callable();
    }
    const auto end = std::chrono::steady_clock::now();

    const double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / static_cast<double>(measured_iterations);
}

inline double giga_operations_per_second(double operations,
                                         double milliseconds) {
    return operations / (milliseconds * 1.0e6);
}

inline double gigabytes_per_second(double bytes, double milliseconds) {
    return bytes / (milliseconds * 1.0e6);
}

inline double million_items_per_second(double items, double milliseconds) {
    return items / (milliseconds * 1.0e3);
}

inline double speedup(double baseline_milliseconds,
                      double candidate_milliseconds) {
    return baseline_milliseconds / candidate_milliseconds;
}

}  // namespace gpu_benchmark
