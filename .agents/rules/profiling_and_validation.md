# Agent Rule: Profiling And Validation

## Purpose

Ensure optimization work stays evidence-driven.

## Rules

- Do not claim a performance win without naming the measurement path.
- Distinguish clearly between:
  - correctness verified
  - build verified
  - performance measured
- Prefer the smallest validation step that proves the requested change.
- For CUDA performance work, start with `./cuda_perf_analysis.sh <binary>` when appropriate.
- When full validation is not possible, state the gap explicitly.

## Review Questions

1. What was actually validated?
2. What remains unverified?
3. Is the reported bottleneck measured or inferred?
