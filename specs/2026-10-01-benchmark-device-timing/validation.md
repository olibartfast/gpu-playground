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
- 2026-10-01, implementation (RTX 3060 Laptop, default preset): `check.sh` exit 0,
  first after T-1..T-3 (run by the T-2 worker) and again after the review fixes (run
  by the orchestrator on the final tree).
  - V-1/V-4: `summarize_check` `PASSED (0 failures)` with host g++ `-Werror`.
  - V-2: `benchmark.h` includes `<algorithm> <chrono> <cmath> <cstdio> <utility> <vector>`.
  - V-3: gemm, softmax, sigmoid and gaussian_blur exit 0 and print `(3 warm-up + 10 timed)`.
  - V-5: `git grep average_milliseconds -- ':!specs'` is empty.
  - V-6: fp16_dot_product, categorical_cross_entropy and gaussian_blur exit 0 with CPU / GPU
    kernel / GPU end-to-end `printBench` lines. fp16 `--performance` passes, with CPU
    `(1 warm-up + 3 timed)`. Gaussian `--performance`: kernel median 0.0635 ms
    (404.6 GFLOP/s), end-to-end 1.25 ms, CPU 127.5 ms.
  - Note: gaussian's kernel median for 10 samples is now the mean of the two middle
    samples (R-1), rather than the upper-middle sample used by the old hand-rolled code.
- V-7: the documentation identifiers were checked against the headers by the T-3 worker
  and by the reviewer.
- V-8 APPROVE (read-only reviewer), no blocking findings. Fixed before commit by a
  corrective packet:
  - N-1: added `<utility>` for `std::move`.
  - N-2: the AGENTS.md snippet declares `kernel_ms` inside the lambda and uses an
    illustrative wrapper.
  - N-3: the cuda-agent-guide copy is replaced by a pointer to AGENTS.md.
  Accepted as-is: N-4 (continuation alignment, a redundant rate label), N-5 (empty
  `summarize` keeps `warmup`, per D-2), N-6 (longer header comments).
- OpenCL builds were not rebuilt locally for this change. `benchmark.h` uses standard
  headers only (V-1/V-2); the PR's OpenCL CI covers both OpenCL backends.
