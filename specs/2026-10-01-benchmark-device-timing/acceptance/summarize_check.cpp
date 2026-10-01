// Acceptance check V-1 / V-4 for specs/2026-10-01-benchmark-device-timing.
// Host-only: compiled with g++ (no nvcc) to prove benchmark.h is backend-agnostic.
// Frozen asset: implementers must not edit this file.
#include "benchmark.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

static int failures = 0;

static void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        failures++;
    }
}

static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    // Odd count, unsorted input.
    BenchResult odd = summarize({3.0, 1.0, 2.0}, 2);
    expect(near(odd.median_ms, 2.0), "odd median");
    expect(near(odd.mean_ms, 2.0), "odd mean");
    expect(near(odd.min_ms, 1.0) && near(odd.max_ms, 3.0), "odd min/max");
    expect(near(odd.stddev_ms, 1.0), "odd sample stddev (n-1)");
    expect(odd.iterations == 3 && odd.warmup == 2, "odd iterations/warmup metadata");

    // Even count.
    BenchResult even = summarize({4.0, 1.0, 3.0, 2.0});
    expect(near(even.median_ms, 2.5), "even median");
    expect(even.warmup == 0, "even default warmup");

    // Single sample.
    BenchResult one = summarize({5.0});
    expect(near(one.median_ms, 5.0) && near(one.stddev_ms, 0.0), "single sample");
    expect(one.iterations == 1, "single iterations");

    // Empty.
    BenchResult none = summarize({});
    expect(none.iterations == 0 && near(none.median_ms, 0.0), "empty vector");

    // benchmarkDevice with defaults: device series is the returned value.
    int calls = 0;
    DeviceBenchResult dev = benchmarkDevice([&] { calls++; return 0.25; });
    expect(near(dev.device.median_ms, 0.25), "device median equals returned value");
    expect(dev.device.iterations == 10 && dev.device.warmup == 3, "device defaults 3+10");
    expect(dev.end_to_end.iterations == 10 && dev.end_to_end.warmup == 3, "end_to_end defaults 3+10");
    expect(calls == 13, "fn called warmup + iterations times");

    // Reset runs before every call; iterations clamp to 1.
    int resets = 0;
    DeviceBenchResult clamped = benchmarkDeviceWithReset(
        [] { return 1.0f; }, [&] { resets++; }, 2, 0);
    expect(clamped.device.iterations == 1, "iterations clamped to 1");
    expect(resets == 3, "reset runs warmup + iterations times");

    // Existing API still works and routes through the same statistics.
    BenchResult legacy = benchmark([] {}, 1, 4);
    expect(legacy.iterations == 4 && legacy.warmup == 1, "benchmark() unchanged");

    // printSpeedup reports a zero candidate median instead of printing nothing.
    BenchResult zero;
    std::fflush(stdout);
    char path[] = "/tmp/summarize_check_XXXXXX";
    int fd = mkstemp(path);
    int saved = dup(1);
    dup2(fd, 1);
    printSpeedup("Speedup:", odd, zero);
    std::fflush(stdout);
    dup2(saved, 1);
    close(saved);
    lseek(fd, 0, SEEK_SET);
    char buf[256] = {};
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    unlink(path);
    expect(n > 0 && std::strstr(buf, "n/a") != nullptr, "printSpeedup n/a on zero median");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
