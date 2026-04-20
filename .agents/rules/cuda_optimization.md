# Agent Rule: CUDA Optimization

## Purpose

Guide CUDA optimization work toward measurable wins without sacrificing correctness or maintainability.

## Rules

- Start with a baseline before tuning.
- Change one optimization variable at a time.
- Use thread counts that are multiples of `32`.
- Treat `128-256` threads per block as a starting point, not a law.
- Prioritize coalesced memory access before advanced micro-optimizations.
- Use shared memory only when reuse clearly offsets synchronization and storage cost.
- Wrap CUDA runtime calls with `CUDA_CHECK`.
- Check kernel launches with `cudaGetLastError()`.
- Avoid device-side allocation.
- Make precision and fast-math tradeoffs explicit.

## Review Questions

1. What bottleneck is this CUDA change targeting?
2. Was correctness checked after the change?
3. Is the performance claim tied to a real measurement?
