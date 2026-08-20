# CUDA Best Practices (CUDA-Agent Style)

This guide distills a practical **analyze → optimize → verify** workflow inspired by the CUDA-Agent project and adapts it to this repository.

Source: *CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation*
([arXiv:2602.24286](https://arxiv.org/abs/2602.24286), [BytedTsinghua-SIA/CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent)).
The priority ordering in §8 and the measurement protocol in §1 are taken from that paper's `SKILL.md` and its
anti-reward-hacking constraints; everything else is adapted to this repository's harness conventions.

## 1) Use a Three-Stage Optimization Loop

### Stage A — Understand and baseline first
- Define the kernel's objective and constraints (throughput vs latency vs memory footprint).
- Build a correctness baseline on CPU or a known-good CUDA reference.
- Measure before tuning: capture wall time, kernel time, and effective bandwidth.

### Stage B — Optimize with one hypothesis at a time
- Change one variable per iteration (block size, memory layout, vector width, etc.).
- Keep changes small and attributable.
- Re-profile after every change.

### Stage C — Verify correctness and regressions
- Compare outputs against a reference within numerical tolerance.
- Add edge-case inputs (tiny tensors, odd dimensions, large dimensions, non-power-of-two sizes).
- Validate against several randomized inputs, not one — a single draw hides boundary bugs.
- Keep a small benchmark table so performance regressions are obvious.

### Measure like a benchmark, not like a stopwatch
A single timed call measures mostly one-off costs: CUDA context creation, OpenCL program build,
first-touch page faults, clock ramp-up. Those can be orders of magnitude larger than the kernel itself,
which makes a lone `steady_clock` pair worse than useless — it looks like data.

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

## 2) Correctness and Safety Rules (Do These Always)

### Always check CUDA runtime calls
```cpp
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err__));                                    \
      std::abort();                                                          \
    }                                                                        \
  } while (0)
```

### Always check launches in debug/development builds
```cpp
myKernel<<<grid, block, 0, stream>>>(...);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaStreamSynchronize(stream));
```

### Keep indexing robust
- Use 64-bit index math when problem sizes can exceed 2^31 elements.
- Guard all global memory accesses with bounds checks.
- Validate shape assumptions early in host code.

## 3) Memory Best Practices

### Prefer coalesced global memory access
- Adjacent threads should load adjacent elements whenever possible.
- Avoid stride-heavy patterns in the innermost access path.

### Use shared memory for reuse, not as a reflex
- Tile only when data is reused enough to offset synchronization overhead.
- Pad shared memory tiles when bank conflicts appear.

### Choose allocation strategy intentionally
- `cudaMalloc/cudaFree`: general-purpose and explicit lifetime.
- `cudaMallocAsync/cudaFreeAsync`: good for stream-ordered allocation patterns.
- `cudaMallocManaged`: productivity-oriented; verify migration overhead under profiling.

### Minimize host-device transfers
- Batch copies.
- Use pinned host memory for frequent transfers.
- Overlap transfers and compute using streams where useful.

## 4) Kernel Configuration and Occupancy

### Start with sane defaults
- Begin with `128-256` threads per block for many scalar kernels.
- Ensure thread count is a multiple of warp size (`32`).

### Tune for your bottleneck
- If memory-bound: improve access patterns and reduce bytes moved.
- If compute-bound: reduce instruction count, increase ILP, and use appropriate intrinsics.
- If occupancy is low: inspect register and shared-memory pressure.

### Use launch bounds only when justified
- `__launch_bounds__` can improve scheduling but can also hurt if set incorrectly.
- Validate launch-bounds changes with profiler data.

## 5) Reduce Warp-Level Waste

- Minimize branch divergence in hot loops.
- Move invariant conditionals outside inner loops.
- Use warp-level primitives (`__shfl_*`, cooperative groups) when they simplify reductions/scans.
- Prefer predication-friendly code paths when possible.

## 6) Numerical Stability and Precision

- Use numerically stable formulations (e.g., softmax max-subtraction).
- Decide precision policy explicitly: FP32, mixed precision, or lower precision.
- Use fast math (`-use_fast_math`) only when accuracy tolerance allows it.
- Validate relative/absolute error against a high-precision reference.

## 7) Practical Profiling Workflow

1. **Nsight Systems**: check launch gaps, CPU-GPU overlap, and stream concurrency.
2. **Nsight Compute**: inspect memory throughput, occupancy, warp stall reasons, and instruction mix.
3. **Compute Sanitizer**: run memcheck/racecheck before and after major rewrites.

Recommended metrics to watch:
- SM utilization
- Achieved occupancy
- DRAM throughput / L2 hit rate
- Warp stall breakdown (memory dependency, execution dependency, barrier, etc.)

## 8) Common Optimization Playbook (In Priority Order)

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

## 9) Code Review Checklist (CUDA-Agent Inspired)

Before merging CUDA changes, verify:
- [ ] Correctness validated against reference outputs, on more than one randomized input.
- [ ] All CUDA API calls and kernel launches are error-checked.
- [ ] Memory access patterns are coalesced in critical loops.
- [ ] Thread-block size chosen from measured data, not guesswork.
- [ ] Register/shared-memory usage reviewed for occupancy impact.
- [ ] Timings come from `benchmark()` (warm-up + repeated iterations), never a single call.
- [ ] Output-as-input kernels reset between benchmark iterations, outside the timed region.
- [ ] Profiling evidence captured (before/after numbers).
- [ ] Edge cases tested (tiny, odd, large, and boundary dimensions).

## 10) Definition of Done for CUDA Optimization

A CUDA optimization is "done" when all are true:
- It is measurably faster on representative workloads.
- It preserves numerical correctness within documented tolerance.
- It does not introduce maintainability hazards (unclear indexing, magic constants without rationale, hidden sync assumptions).
- It includes enough benchmark context for future regression checks.
