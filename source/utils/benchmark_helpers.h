#pragma once

// Converts a latency (normally a BenchResult::median_ms from benchmark.h)
// into a rate: throughput, bandwidth, or a speedup ratio. Timing itself lives
// in benchmark.h (benchmark(), benchmarkWithReset(), benchmarkDevice(),
// benchmarkDeviceWithReset()); this file only derives reportable numbers from
// the milliseconds those produce.

namespace gpu_benchmark {

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
