# CUDA Agent Guide

This is the consolidated CUDA optimization guide for agentic work in this repo. It replaces the older split between `cuda-copilot-rules.md` and `cuda_best_practice_agent_style.md`.

The priority ordering in the optimization playbook and the measurement protocol below are taken from
*CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation*
([arXiv:2602.24286](https://arxiv.org/abs/2602.24286),
[BytedTsinghua-SIA/CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent)) and its
anti-reward-hacking constraints; everything else is adapted to this repository's harness conventions.

## Operating Loop

### 1. Understand and baseline first

- Define the goal: throughput, latency, memory footprint, or numerical stability.
- Read the target kernel and its harness before changing code.
- Capture a baseline with the current binary before tuning.

### 2. Optimize one hypothesis at a time

- Change one variable per iteration:
  - block size
  - memory layout
  - vector width
  - shared-memory tiling
  - instruction mix
- Keep edits small enough that wins and regressions stay attributable.

### 3. Verify after every meaningful change

- Compare against a CPU path or known-good output, on more than one randomized input —
  a single draw hides boundary bugs.
- Test small, odd, and boundary-sized inputs.
- Re-profile after each optimization pass.

## Measure Like A Benchmark, Not Like A Stopwatch

A single timed call measures mostly one-off costs: CUDA context creation, first-touch page faults,
clock ramp-up. Those can be orders of magnitude larger than the kernel itself, which makes a lone
`steady_clock` pair worse than useless — it looks like data.

The protocol every harness in this repository uses:

1. Run several **untimed warm-up calls** and discard them.
2. Time **N further iterations** and report the distribution, not one number.
3. Compare on the **median**; the mean is the statistic a single stalled iteration moves most.
4. Report the spread (stddev, min/max) so a noisy measurement is visible instead of averaged away.
5. If the output buffer is also an input (GEMM's `beta * C`, any in-place kernel), **reset it between
   iterations** and keep the reset out of the timed region — otherwise each iteration measures
   different work and the post-benchmark correctness check compares against garbage.

`source/utils/benchmark.h` implements this and is available to all three backends:

```cpp
#include "benchmark.h"

BenchResult cpu_bench = benchmark([&] { softmax_cpu(input, out_cpu, N); });
BenchResult gpu_bench = benchmark([&] { softmax_gpu(input, out_gpu, N); });
printBench("CPU:", cpu_bench);
printBench("GPU:", gpu_bench);
printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

// Output doubles as input: reset before every call, untimed.
BenchResult gemm_bench = benchmarkWithReset(
    [&] { gemmGPU(A, B, C, alpha, beta, M, N, K); },
    [&] { std::copy(C_seed, C_seed + M * N, C); });
```

Defaults are 3 warm-up + 10 timed iterations; pass explicit counts as trailing arguments when a
reference implementation is slow enough that the full protocol dominates the harness runtime.

## Hard CUDA Rules

- Use a thread block size that is a multiple of `32`.
- Start with `128-256` threads per block unless measurements suggest otherwise.
- Wrap all CUDA runtime calls with `CUDA_CHECK`.
- Check kernel launches with `cudaGetLastError()`.
- Guard global memory accesses with bounds checks.
- Do not allocate memory inside kernels.
- Use 64-bit indexing when element counts can exceed 32-bit ranges.

## Memory Rules

- Prefer coalesced global memory access.
- Use shared memory for reuse, not by reflex.
- Pad shared-memory tiles if bank conflicts appear.
- Prefer structure-of-arrays layouts when they improve coalescing.
- Minimize host-device transfers and batch them where possible.
- Use pinned host memory for frequent transfer paths.

## Occupancy And Scheduling

- Ensure grid size is large enough to keep SMs busy.
- Treat occupancy as a constraint, not the final objective.
- Inspect register pressure and shared-memory use before forcing launch bounds.
- Use `__launch_bounds__` only with profiler evidence.

## Divergence And Warp Efficiency

- Avoid divergent branches in hot loops.
- Move invariant conditionals out of inner loops.
- Use warp-level primitives when they simplify reductions or scans.
- Prefer predication-friendly code when it reduces warp waste.

## Numerical Policy

- Use stable formulations for reductions and normalization paths.
- Make precision policy explicit: FP32, mixed precision, or lower precision.
- Use fast math only when the accuracy budget is clear.

## Practical Profiling Order

1. Time CPU/GPU harnesses with `benchmark()` / `benchmarkWithReset()` from
   `benchmark.h`; use `benchmark_helpers.h` to turn those latencies into
   throughput, bandwidth, or speedup figures. State whether GPU timing is
   kernel-only or end-to-end.
2. Use `./cuda_perf_analysis.sh <binary>` for a first profiler pass.
3. Use Nsight Systems for launch gaps and stream overlap.
4. Use Nsight Compute for throughput, occupancy, stalls, and instruction mix.
5. Use Compute Sanitizer before and after major rewrites.

## Optimization Playbook (In Priority Order)

Work top-down. Exhausting a tier before descending is what keeps effort proportional to payoff —
most kernels never need tier 3, and reaching for it first is the classic way to spend a day for 4%.

### Priority 0 — Algebraic simplification (unbounded impact)

Before optimizing the computation, check whether it is the right computation.

- Recognize implicit structure: multiplying by a diagonal matrix is row scaling, not a GEMM.
- Exploit linearity to reorder reductions: `sum(x · Wᵀ)` becomes `x · sum(Wᵀ)`.
- Eliminate materialized intermediates that exist only to satisfy a generic operator's signature.

This tier changes asymptotic complexity, so it dominates everything below it when it applies.

### Priority 1 — Algorithmic and memory structure (>50% impact)

- Kernel fusion — collapse chains of operations to cut global-memory round trips and launch overhead.
- Shared-memory tiling — only where data reuse justifies the synchronization cost.
- Memory coalescing — adjacent threads touching adjacent addresses in the hot path.

### Priority 2 — Hardware utilization (20-50% impact)

- Vectorized loads/stores (`float2`/`float4`), only when alignment is guaranteed.
- Warp-level primitives (`__shfl_sync`, `__ballot_sync`) for intra-warp reductions and scans.
- Occupancy tuning via block size and register pressure.

### Priority 3 — Fine-tuning (<20% impact)

- Instruction-level parallelism and loop unrolling.
- Mixed precision (FP16/TF32) where the accuracy budget allows — this is also how you reach Tensor Cores.
- Prefetching and double buffering.

### Priority 4 — Parameter sweeps (last resort)

Only once you are within ~1.2x of the target and the tiers above are genuinely exhausted.
Template the kernel on its tuning parameters and dispatch at runtime so a sweep needs no recompile.

### Always available: use the vendor library

cuBLAS for GEMM and cuDNN for convolution are mature and hardware-tuned. Hand-written kernels here are
a *learning* exercise; when the goal is speed rather than understanding, mapping onto a fused library
primitive usually wins and should be the measured baseline either way.

### Per-iteration discipline

- State the expected win before making the change ("fusion removes 3 launches, expect ~20%").
- Measure, then explain the gap between expectation and result — that gap is where the learning is.
- Do not revert to a slower version to make a correctness bug easier to fix; fix it in the optimized one.

## Review Checklist

- [ ] Correctness checked against a reference path, on more than one randomized input
- [ ] CUDA calls and launches are error-checked
- [ ] Memory access patterns are coalesced in hot paths
- [ ] Block size is justified by measurement, not habit
- [ ] Register and shared-memory pressure were considered
- [ ] Timings come from `benchmark()` (warm-up + repeated iterations), never a single call
- [ ] Output-as-input kernels reset between benchmark iterations, outside the timed region
- [ ] Performance claims include enough context to reproduce
