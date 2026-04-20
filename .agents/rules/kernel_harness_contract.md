# Agent Rule: Kernel Harness Contract

## Purpose

Keep per-kernel harnesses consistent so kernels remain easy to build, compare, and extend.

## Rules

- Preserve the pattern where `main.cpp` drives CPU reference and GPU execution.
- Keep allocation, transfers, and kernel launch code inside backend implementation files, not the harness.
- Return non-zero on validation failure.
- Keep the harness readable and focused on setup, invocation, timing, and comparison.
- When adding a new kernel, keep naming and directory structure aligned with existing kernels.

## Review Questions

1. Does `main.cpp` still behave as a harness instead of becoming a backend implementation file?
2. Are CPU reference and GPU paths still comparable?
3. Did the change preserve the repo's per-kernel template shape?
