# CUDA Agent Guide

This is the consolidated CUDA optimization guide for agentic work in this repo. It replaces the older split between `cuda-copilot-rules.md` and `cuda_best_practice_agent_style.md`.

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

- Compare against a CPU path or known-good output.
- Test small, odd, and boundary-sized inputs.
- Re-profile after each optimization pass.

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

1. Use `./cuda_perf_analysis.sh <binary>` for a first pass.
2. Use Nsight Systems for launch gaps and stream overlap.
3. Use Nsight Compute for throughput, occupancy, stalls, and instruction mix.
4. Use Compute Sanitizer before and after major rewrites.

## Review Checklist

- [ ] Correctness checked against a reference path
- [ ] CUDA calls and launches are error-checked
- [ ] Memory access patterns are coalesced in hot paths
- [ ] Block size is justified by measurement, not habit
- [ ] Register and shared-memory pressure were considered
- [ ] Performance claims include enough context to reproduce
