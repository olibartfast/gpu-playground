# Agent Rule: Backend Boundaries

## Purpose

Preserve the repository's backend split and keep implementation details in the correct place.

## Rules

- Keep CUDA code in `source/<kernel>/cuda/`.
- Keep OpenCL C API code in `source/<kernel>/opencl/`.
- Keep OpenCL C++ wrapper code in `source/<kernel>/opencl_cpp/`.
- Keep `main.cpp` as the backend-agnostic harness entrypoint.
- Do not leak CUDA or OpenCL-specific types into backend-agnostic headers unless the existing file already does that.
- Do not mix backend plumbing into the harness when it belongs in the backend implementation file.

## Review Questions

1. Does the change preserve the three-surface backend layout?
2. Are backend-specific includes and resource lifetimes kept in the right directory?
3. Would another backend still compile cleanly after this change?
