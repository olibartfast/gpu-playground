# OpenCL Agent Guide

This is the consolidated OpenCL optimization guide for agentic work in this repo.

## Operating Loop

### 1. Understand and baseline first

- Define the target metric: throughput, latency, or portability.
- Start from a correctness baseline on CPU or a known-good kernel.
- Measure wall time, event time, and effective bandwidth before tuning.

### 2. Optimize one hypothesis at a time

- Change one variable per pass:
  - local size
  - memory layout
  - vector width
  - local-memory tiling
  - build options
- Keep the diff attributable.

### 3. Verify and regression-check

- Compare outputs against a reference within tolerance.
- Test tiny, odd, and non-multiple-of-local-size shapes.
- Keep simple benchmark notes by device and driver when performance matters.

## Host API Rules

- Prefer the Khronos C++ wrapper when the host code is C++.
- If using the C API, check every return value explicitly.
- Always retrieve and inspect build logs on program compilation failures.
- Query device capabilities up front instead of assuming OpenCL 2.x or 3.0 optional features exist.

## In-Kernel Portability Rules

- Guard optional features with `__opencl_c_*` feature macros.
- Do not assume support for:
  - sub-groups
  - FP16
  - FP64
  - SVM
- Provide a fallback path or fail early with a clear error when the feature is required.

## Memory Rules

- Prefer coalesced global memory access.
- Use `__local` memory only when reuse clearly outweighs barrier and storage overhead.
- Re-test `__local` optimizations on the actual device class you care about; desktop and mobile behavior differs.
- Use `__constant` memory for broadcast-style read-mostly values when it fits the device.

## Work-Group Rules

- Start from `64-256` work-items per work-group for many scalar kernels.
- Query `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`.
- Keep local size within both kernel and device limits.
- Tune according to the real bottleneck:
  - access pattern for memory-bound kernels
  - instruction mix for compute-bound kernels
  - register and local-memory pressure for low-residency kernels

## Practical Profiling

- Use OpenCL event profiling first.
- Use vendor profilers for memory throughput, occupancy or residency, cache behavior, and stalls.
- Use tools such as Oclgrind when debugging memory or synchronization issues.

## Review Checklist

- [ ] Correctness checked against a reference path
- [ ] API calls and build failures are handled explicitly
- [ ] Optional OpenCL features are queried on the host
- [ ] In-kernel feature guards are present where needed
- [ ] Memory access patterns are sane in hot loops
- [ ] Local size is justified by data
- [ ] Device and driver context are recorded for performance claims
