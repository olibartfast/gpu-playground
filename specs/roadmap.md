# Complexity-Study Roadmap

This roadmap tracks the planning and delivery of the repository's documentation
and benchmarking work. Each section names its specification, branch, and phase
status. Existing migration and LoRA plans retain their own scope and are not
superseded by this roadmap.

## Kernel Complexity

Specification: [kernel complexity](2026-09-20-kernel-complexity/plan.md).
Branch: `docs/kernel-complexity-plan`.

| Phase | Outcome | Status |
|---|---|---|
| 0 | Requirements, phased plan, and validation criteria | Drafted; planning checks recorded in `validation.md` |
| 1 | Common cost model and three source-grounded pilot studies | Planned |
| 2 | A study for every remaining implemented kernel and backend variant | Planned |
| 3 | Scaling evidence for representative families and ranked opportunities | Planned |
| 4 | One independently validated optimization at a time | Deferred to separate implementation packets |

Competition submissions and external framework examples are outside this
increment. Reconsider them after completing coverage of `source/`.

## Benchmark Framework

Specification: [device-time
benchmarking](2026-10-01-benchmark-device-timing/requirements.md).
Branch: `feat/benchmark-device-timing`.

| Phase | Outcome | Status |
|---|---|---|
| Pre | `gaussian_blur` coherent and merged to `master` ([spec](2026-10-01-gaussian-blur/requirements.md), merged in PR #6) | Done 2026-10-01 |
| 0 | Requirements, plan, validation, and frozen acceptance | Done 2026-10-01 |
| 1 | `summarize` and `benchmarkDevice` in `benchmark.h` | Done 2026-10-01 |
| 2 | Retire `average_milliseconds`; migrate three harnesses (including `gaussian_blur`); update docs | Done 2026-10-01 |
| Deferred | CSV/JSON output, time-budget iterations, percentile reporting | Idea |

Complexity Phase 3 (task T-6) can use `benchmarkDevice` for kernel-only
boundaries once Phase 1 lands.

## Python DSL Backends

Specification: [Python backends](2026-10-01-python-backend/requirements.md).
Branch: `feat/python-backend`.

| Phase | Outcome | Status |
|---|---|---|
| 0 | Requirements, plan, validation, and frozen acceptance | Done 2026-10-01 |
| 1 | `source/utils/python/gpu_bench.py` and requirements | Done 2026-10-01 |
| 2 | Migrate the three existing Triton/CuTe scripts; update docs | Done 2026-10-01 (CuTe execution gap: `cutlass` not installed) |
| Idea | GPU-less `py_compile` CI job; Python ports of more kernels; CuTe DSL environment | Idea |

## HIP / AMD Backend

Specification: [HIP backend
foundation](2026-10-19-hip-backend/requirements.md).
Branch: `feat/hip-backend` (proposed; not created).

| Phase | Outcome | Status |
|---|---|---|
| 0 | Requirements, plan, validation, and scope freeze | Done 2026-10-19 |
| 1 | `USE_HIP` build path, `hip_helpers`, `hip` preset, `gfx942` configure | Planned |
| 2 | `sigmoid` HIP backend validated on MI300X | Planned |
| 3 | Wave64 policy, `categorical_cross_entropy` HIP backend, device timing | Planned |
| 4 | Documentation, CUDA/OpenCL regression, integration | Planned |
| Future | Remaining portable kernels, GEMM via rocWMMA/MFMA, cooperative groups, CI, `gfx950`/RDNA | Deferred |
