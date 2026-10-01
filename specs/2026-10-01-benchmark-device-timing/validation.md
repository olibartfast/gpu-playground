# Validation specified before implementation

Frozen acceptance command (run from the repo root; exits nonzero on any failure):

```bash
specs/2026-10-01-benchmark-device-timing/acceptance/check.sh
```

| Check | Requirements | Observable evidence |
|---|---|---|
| V-1 | R-1, R-2, R-3 | `acceptance/summarize_check.cpp` compiles with `g++ -std=c++17 -Wall -Wextra -Werror -I source/utils` (no nvcc) and passes. It asserts: odd count, even count, single sample and empty vector for `summarize`; `benchmarkDevice` with a constant-returning lambda gives `device.median_ms` == constant, `iterations == 10`, `warmup == 3`; clamping of `iterations = 0` → 1; the reset runs warmup+iterations times |
| V-2 | R-3 | `grep '#include' source/utils/benchmark.h` lists only standard headers |
| V-3 | R-4 | Default preset builds; `gemm`, `softmax`, `sigmoid` exit 0 and print `(3 warm-up + 10 timed)` lines |
| V-4 | R-5, R-6 | Header contains the sync-contract comment. `summarize_check` captures the `printSpeedup` output for a zero candidate median and finds `n/a` |
| V-5 | R-7 | `git grep -n average_milliseconds -- ':!specs'` returns nothing |
| V-6 | R-8 | `fp16_dot_product` and `categorical_cross_entropy` exit 0 on functional tests; their output contains `GPU kernel` and `GPU end-to-end` lines with `(3 warm-up + 10 timed)`. `fp16_dot_product --performance` exits 0. `gaussian_blur` and `gaussian_blur --performance` exit 0, and the `--performance` output has `GPU kernel` and `GPU end-to-end` `printBench` lines |
| V-7 | R-9 | V-5 grep plus manual reading: every API name in the docs matches R-1/R-2 exactly |
| V-8 | all | Read-only reviewer verdict on the full diff against this packet; changed paths are limited to those owned in plan.md |

OpenCL builds (`USE_OPENCL`, `USE_OPENCL_CPP`) are rebuilt when an OpenCL SDK is
available. Otherwise the gap is recorded here. V-1's host-only compile is the
guaranteed evidence for R-3.

## Evidence

- 2026-10-01, T-0 (acceptance assets frozen): `bash -n check.sh` passes. On the
  pre-implementation tree, `check.sh` exits 1 at "V-1 compile" (`summarize`,
  `benchmarkDevice` undeclared), as expected. `summarize_check.cpp` compiled with
  host g++ against the stashed pre-spec header prints `PASSED (0 failures)`. This
  shows the checks can pass. It is not evidence for the implementation, which
  restarts from T-1.
- V-1..V-8 against the implementation: not started.
