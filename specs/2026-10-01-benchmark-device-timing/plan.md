# Plan: device-time benchmarking

Requirements: [requirements.md](requirements.md). Checks are defined in
[validation.md](validation.md) before implementation. Execution follows
[implementation-workflow.md](implementation-workflow.md).

## Prerequisite

`feat/gaussian-blur` is merged to `master` (its own packet), then this branch is
rebased onto `master`. No dispatch before that.

## Phase 0 — Acceptance assets (specifier, before any dispatch)

- **T-0 (V-all):** freeze `acceptance/check.sh` and `acceptance/summarize_check.cpp`
  in this packet. Record the exact invocation in validation.md. Workers never
  edit this directory.

## Phase 1 — Framework (walk)

- **T-1 (R-1–R-6):** edit `source/utils/benchmark.h` only:
  - add `summarize`, `DeviceBenchResult`, `benchmarkDevice` and
    `benchmarkDeviceWithReset`;
  - route `benchmarkWithReset` through `summarize`;
  - add the sync-contract comment and the `printSpeedup` n/a line.

Gate: V-1, V-2, V-3, V-4. Existing harnesses unchanged and passing.

## Phase 2 — Retire the second framework (run)

- **T-2 (R-7, R-8):** owns `source/utils/benchmark_helpers.h`,
  `source/fp16_dot_product/main.cpp`, `source/categorical_cross_entropy/main.cpp`
  and `source/gaussian_blur/main.cpp`. Remove `average_milliseconds` and migrate
  all three harnesses.
- **T-3 (R-9):** owns the six docs files listed in R-9. It can run in parallel
  with T-2 because the API is frozen by R-1/R-2 and the file sets are disjoint.

Gate: V-5, V-6, V-7, then full acceptance (V-all) and independent review (V-8).

## Integration

One PR from `feat/benchmark-device-timing` containing this packet, the code and the docs.
On merge, update [roadmap](../roadmap.md) status and record validation evidence.
