# OpenCL Optimization Guide

> Audience: agents · Applies to: OpenCL · Last reviewed: 2026-10

This is the working guide for OpenCL optimization in this repository. It covers
host API rules, portability constraints, work-group sizing, measurement, and
profiling. For staged OpenCL 3.1 work and its completion criteria, see the
[OpenCL 3.1 migration plan](../../specs/2026-05-28-opencl-3.1-migration/plan.md).

## Operating Loop

### 1. Understand and Baseline First

- Define the target metric: throughput, latency, or portability.
- Start from a correctness baseline on CPU or a known-good kernel.
- Measure wall time, event time, and effective bandwidth before tuning.

### 2. Optimize One Hypothesis at a Time

Change one variable per pass:

- local size
- memory layout
- vector width
- local-memory tiling
- build options

Keep the diff attributable.

### 3. Verify and Regression-Check

- Compare outputs against a reference within tolerance.
- Test tiny, odd, and non-multiple-of-local-size shapes.
- Keep simple benchmark notes by device and driver when performance matters.

## Host API Rules

- Prefer the Khronos C++ wrapper when the host code is C++.
- If you use the C API, check every return value explicitly.
- Always retrieve and inspect build logs when program compilation fails.
- Query device capabilities up front instead of assuming that OpenCL 2.x or 3.0
  optional features exist.

## In-Kernel Portability Rules

- Guard optional features with `__opencl_c_*` feature macros.
- Do not assume support for:
  - sub-groups
  - FP16
  - FP64
  - SVM
- Provide a fallback path, or fail early with a clear error when the feature is
  required.

## Memory Rules

- Prefer coalesced global memory access.
- Use `__local` memory only when reuse clearly outweighs barrier and storage
  overhead.
- Re-test `__local` optimizations on the actual device class you care about;
  desktop and mobile behavior differs.
- Use `__constant` memory for broadcast-style, read-mostly values when it fits
  the device.

## Work-Group Rules

- Start with `64–256` work-items per work-group for many scalar kernels.
- Query `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`.
- Keep the local size within both kernel and device limits.
- Tune according to the real bottleneck:
  - access pattern for memory-bound kernels
  - instruction mix for compute-bound kernels
  - register and local-memory pressure for low-residency kernels

## Measure Like a Benchmark, Not a Stopwatch

The first call into an OpenCL path pays context creation and `clBuildProgram`,
which routinely costs more than the kernel it is about to run. Timing one call
therefore measures the build, not the kernel.

Use `source/utils/benchmark.h` — it is backend-agnostic and pulls in no OpenCL
headers. It runs several untimed warm-up calls, then repeated timed iterations
reported as median plus spread. Compare on medians. If the output buffer is also
an input, pass a `reset` to `benchmarkWithReset()` so every iteration starts
from the same state, with the reset excluded from the timed region.

See [the CUDA guide](cuda.md) for the full protocol and examples.

## Practical Profiling

- Start with OpenCL event profiling.
- Use vendor profilers for memory throughput, occupancy or residency, cache
  behavior, and stalls.
- Use tools such as Oclgrind when debugging memory or synchronization issues.

## Review Checklist

- [ ] Correctness checked against a reference path, on more than one randomized
      input
- [ ] API calls and build failures are handled explicitly
- [ ] Optional OpenCL features are queried on the host
- [ ] In-kernel feature guards are present where needed
- [ ] Memory access patterns are sane in hot loops
- [ ] Local size is justified by data
- [ ] Timings come from `benchmark()` (warm-up and repeated iterations), never a
      single call
- [ ] Device and driver context are recorded for performance claims
