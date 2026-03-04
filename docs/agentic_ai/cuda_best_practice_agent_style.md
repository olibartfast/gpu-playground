# CUDA Best Practices (CUDA-Agent Style)

This guide distills a practical **analyze → optimize → verify** workflow inspired by the CUDA-Agent project and adapts it to this repository.

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
- Keep a small benchmark table so performance regressions are obvious.

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

## 8) Common Optimization Playbook

- Fuse kernels when global-memory round trips dominate.
- Vectorize loads/stores (`float2/float4`) only when alignment is guaranteed.
- Hoist repeated computations out of loops.
- Precompute constants and use constant memory for broadcast-style reads.
- Replace expensive operations with equivalent cheaper forms when numerically acceptable.

## 9) Code Review Checklist (CUDA-Agent Inspired)

Before merging CUDA changes, verify:
- [ ] Correctness validated against reference outputs.
- [ ] All CUDA API calls and kernel launches are error-checked.
- [ ] Memory access patterns are coalesced in critical loops.
- [ ] Thread-block size chosen from measured data, not guesswork.
- [ ] Register/shared-memory usage reviewed for occupancy impact.
- [ ] Profiling evidence captured (before/after numbers).
- [ ] Edge cases tested (tiny, odd, large, and boundary dimensions).

## 10) Definition of Done for CUDA Optimization

A CUDA optimization is "done" when all are true:
- It is measurably faster on representative workloads.
- It preserves numerical correctness within documented tolerance.
- It does not introduce maintainability hazards (unclear indexing, magic constants without rationale, hidden sync assumptions).
- It includes enough benchmark context for future regression checks.
