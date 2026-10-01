# Requirements: gaussian_blur kernel, coherent with the CUDA examples

Created: 2026-10-01. Status: specified; coherence changes not started.
Branch: `feat/gaussian-blur` (commit 17deb2d adds the kernel). This packet was
written after that commit. It governs the changes needed before merging to `master`.

Constraints: [mission](../mission.md), [technical boundaries](../tech-stack.md),
[repository contract](../../AGENTS.md), [new-kernel checklist](../../docs/adding-a-new-kernel.md#checklist).

## Findings (2026-10-01, branch vs. the other 18 harnesses)

- F-1: `source/gaussian_blur/` uses 2-space indent and `float *p`. Every CUDA
  example except `categorical_cross_entropy` uses 4-space indent, and the
  closest sibling, `convolution2d`, uses `float* p`.
- F-2: `main.cpp` includes `cuda/gaussian_blur.h` directly. 14 of 19 harnesses,
  including `convolution2d`, use the checklist's 3-way `GPU_OPENCL_CPP_BACKEND` /
  `GPU_OPENCL_BACKEND` / CUDA include guard.
- F-3: functional tests call the CPU and GPU paths once, untimed. The checklist
  requires timings from `benchmark()`.
- F-4: `main.cpp` ends with a LeetGPU-specific message rather than the
  `Overall result: PASSED/FAILED` line used by `convolution2d`, `gemm` and `spmv`.
- F-5: the `__global__` kernel has external linkage. `convolution2d` marks its
  kernels `static __global__`.
- F-6: the `--performance` path hand-rolls a kernel-time median. This is
  deliberately left for the benchmark feature (see out of scope).
- F-7: correctness (5 functional tests plus `--performance`) passed at 17deb2d
  on an RTX 3060 Laptop (session record, 2026-10-01).

## Requirements

- **R-1 — Style:** reformat `cuda/gaussian_blur.{h,cpp}` and `main.cpp` to
  4-space indent with `type* name` pointers, matching `convolution2d`.
  Formatting only: the kernel body's logic stays exactly as the user wrote it.
- **R-2 — Include guard:** `main.cpp` uses the 3-way backend include guard
  (`#include "opencl_cpp/gaussian_blur.h"` / `"opencl/gaussian_blur.h"` /
  `"cuda/gaussian_blur.h"`). The root CMake gate still keeps the target CUDA-only,
  so the OpenCL branches are never compiled.
- **R-3 — Timed functional tests:** each functional test runs the CPU reference
  and `gaussian_blur_gpu` through `benchmark()` and prints them with `printBench`
  (`CPU:`, `GPU:`) and `printSpeedup`. Validation uses the outputs of the
  benchmarked calls.
- **R-4 — Result line:** `main.cpp` prints `Overall result: PASSED` or
  `Overall result: FAILED`, in both the functional and `--performance` modes, and
  returns 0 or 1 accordingly.
- **R-5 — Kernel linkage:** `static __global__ void gaussian_blur(...)`.
  `extern "C" solve` stays unchanged: the LeetGPU entrypoint, as in
  `categorical_cross_entropy`.
- **R-6 — Contract preserved:**
  - The test cases, the LeetGPU tolerances (atol = rtol = 1e-5), the `--performance`
    shape (512x512, 7x7), the public header signatures and the Readme row stay the same.
  - The tolerance is a deliberate, documented deviation from the repo's ~1e-4 default,
    because it is the challenge's contract.

## Out of scope

- F-6: moving `--performance` onto `benchmarkDevice()`. That belongs to
  `specs/2026-10-01-benchmark-device-timing` (on `feat/benchmark-device-timing`),
  which lands after this merge.
- An OpenCL port, and any optimization (tiling or shared memory) of the naive kernel.
- Restyling `categorical_cross_entropy` (the other 2-space outlier).
- The complexity-study entry for `gaussian_blur`. `docs/algorithmic-complexity.md`
  lives on the unmerged `feat/complexity-studies`; whichever of the two merges
  second adds the entry.

## Decisions

- D-1: "Coherent" means the new-kernel checklist plus the majority harness
  convention, with `convolution2d` as the reference sibling. Where the LeetGPU
  contract conflicts (tolerances, `solve`), the contract wins and is documented.
- D-2: merge through a PR to `master`, as for PRs #1–#5.
