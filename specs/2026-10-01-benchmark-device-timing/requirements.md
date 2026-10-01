# Requirements: device-time benchmarking and one timing framework

Created: 2026-10-01. Status: implemented and validated 2026-10-01 (see validation.md).
Branch: `feat/benchmark-device-timing`, rebased onto `master` after
`feat/gaussian-blur` merges (see `specs/2026-10-01-gaussian-blur/`). Unblocked 2026-10-01:
PR #6 merged (8b68407); this branch fast-forwarded onto it.

## Goal and scope

Make `source/utils/benchmark.h` the only timing framework, with first-class
kernel-only timing that reuses its warm-up / median / spread protocol.

Constraints: [mission](../mission.md), [technical boundaries](../tech-stack.md),
[repository contract](../../AGENTS.md).

## Findings (2026-10-01 source review)

- F-1: `benchmark()` / `benchmarkWithReset()` time a host callable with
  `steady_clock`. GPU wrappers allocate, copy, launch, copy back and free inside
  the timed call (e.g. `source/sigmoid/cuda/sigmoid.cpp`), so "GPU" timings
  are end-to-end.
- F-2: `fp16_dot_product` and `categorical_cross_entropy` return kernel-only time
  through a `float* kernel_time_ms` out-parameter (CUDA events). The framework
  cannot summarize these samples.
- F-3: `gpu_benchmark::average_milliseconds` (`source/utils/benchmark_helpers.h`)
  reports a mean only, timing the whole loop. Both harnesses in F-2 call it with
  0 warm-up / 1 iteration, which is the single-timed-call pattern AGENTS.md forbids.
  `docs/EXAMPLES.md` still recommends it.
- F-4: the requirement that a timed callable block until device work completes
  is documented only in `benchmark_helpers.h`.
- F-5: `printSpeedup` prints nothing when the candidate median is 0.
- F-6: `gaussian_blur --performance` (merged first per the user's direction)
  collects `kernel_time_ms` samples alongside `benchmark()`, then drops the
  warm-up samples and takes the median by hand.

## Requirements

- **R-1 — Shared statistics:** a public `summarize(std::vector<double> samples,
  int warmup = 0)` returns a `BenchResult`. It matches today's math: the median
  is the middle element for an odd count and the mean of the two middle elements
  for an even count; stddev is the sample stddev (n-1), or 0 for a single sample.
  An empty input yields a zeroed result with `iterations == 0`.
  `benchmarkWithReset` computes its result through `summarize`.
- **R-2 — Device timing:** `benchmarkDevice(fn, warmup, iterations)` and
  `benchmarkDeviceWithReset(fn, reset, warmup, iterations)` return
  `DeviceBenchResult { BenchResult end_to_end; BenchResult device; }`.
  `fn` returns that call's device-measured milliseconds, convertible to double.
  Warm-up calls are discarded for both series. `reset` is untimed and runs
  before every call. Defaults and clamping (`iterations < 1`, `warmup < 0`)
  match `benchmarkWithReset`.
- **R-3 — Backend-agnostic:** `benchmark.h` includes standard headers only.
  It compiles with a host C++17 compiler alone and in all three backend builds.
- **R-4 — Compatibility:** existing `benchmark()` / `benchmarkWithReset()` /
  `printBench` call sites compile unchanged and keep the same output format.
- **R-5 — Sync contract:** the `benchmark.h` header comment states that a timed
  callable must block until its device work completes; a blocking
  device-to-host copy counts.
- **R-6 — Visible speedup failure:** `printSpeedup` prints
  `<label> n/a (candidate median is 0)` when the candidate median is not positive.
- **R-7 — Single framework:** remove `gpu_benchmark::average_milliseconds`.
  `benchmark_helpers.h` keeps only the unit converters (`giga_operations_per_second`,
  `gigabytes_per_second`, `million_items_per_second`, `speedup`), documented
  as operating on `median_ms`.
- **R-8 — Harness migration:** in `fp16_dot_product`,
  `categorical_cross_entropy` and the `gaussian_blur --performance` path:
  - The CPU path uses `benchmark()` and the GPU path uses `benchmarkDevice()`.
  - Timings print via `printBench` (CPU, GPU kernel, GPU end-to-end), and rates
    and speedups are computed from medians.
  - Test cases, tolerances, pass/fail logic and exit codes are unchanged.
  - `fp16_dot_product --performance` (n = 1e8) passes explicit short counts
    (`1, 3`) to the CPU benchmark.
  - `gaussian_blur`'s hand-rolled median is replaced by `benchmarkDevice()`; its
    GFLOP/s and GB/s lines use `device.median_ms`.
- **R-9 — Docs:** bring these into line with R-1–R-7, with no remaining
  `average_milliseconds` reference:
  - `AGENTS.md` "Testing And Benchmarking"
  - `docs/EXAMPLES.md`
  - `docs/cuda-agent-guide.md`
  - `docs/adding-a-new-kernel.md`
  - `Readme.md`
  - `specs/tech-stack.md`

## Out of scope

- CSV/JSON output, time-budget iteration modes and percentile/IQR reporting
  are deferred in the [roadmap](../roadmap.md).
- Any GPU wrapper, kernel or OpenCL host change. OpenCL event timing through
  `benchmarkDevice()` is possible later but not part of this increment.

## Decisions and assumptions

- D-1: device time comes from the callable's return value rather than from a
  CUDA/OpenCL timer inside the header. This keeps R-3 and lets each backend
  measure its own way.
- D-2: `summarize` takes samples that already exclude warm-up. `warmup` is
  metadata for `printBench` only.
- A-1: 3 warm-up + 10 timed iterations stay the defaults.
- Pre-spec implementation attempt (2026-10-01): P1/P3 agents started before
  this packet existed and were stopped. Their edits are preserved unreviewed in
  `git stash` ("benchmark-device-timing: premature P1+P3 implementation
  (pre-spec)"). They are reference material only; implementation restarts from
  this packet.
