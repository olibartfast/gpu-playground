# Complexity-study roadmap

Scope: this requested study only; existing migration and LoRA plans retain their scope.

Specification: [kernel complexity](2026-09-20-kernel-complexity/plan.md).
Branch: `docs/kernel-complexity-plan`.

| Phase | Outcome | Status |
|---|---|---|
| 0 | Requirements, phased plan, and validation criteria | Drafted; planning checks recorded in validation.md |
| 1 | Common cost model and three source-grounded pilot studies | Planned |
| 2 | Study for every remaining implemented source kernel and backend variant | Planned |
| 3 | Scaling evidence for representative families and ranked opportunities | Planned |
| 4 | One independently validated optimization at a time | Deferred; separate implementation packets |

Competition submissions and external framework examples are outside this increment.
Reconsider them after completing coverage of `source/`.

## Benchmark framework

Specification: [device-time benchmarking](2026-10-01-benchmark-device-timing/requirements.md).
Branch: `feat/benchmark-device-timing`.

| Phase | Outcome | Status |
|---|---|---|
| Pre | `gaussian_blur` coherent and merged to `master` ([spec](2026-10-01-gaussian-blur/requirements.md), merged in PR #6) | Done 2026-10-01 |
| 0 | Requirements, plan, validation, frozen acceptance | Specified 2026-10-01 |
| 1 | `summarize` + `benchmarkDevice` in `benchmark.h` | Planned |
| 2 | Retire `average_milliseconds`; migrate three harnesses (incl. `gaussian_blur`); docs | Planned |
| Deferred | CSV/JSON output, time-budget iterations, percentile reporting | Idea |

Complexity Phase 3 (T-6) can use `benchmarkDevice` for kernel-only boundaries once Phase 1 lands.
