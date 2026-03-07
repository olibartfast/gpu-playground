# OpenCL Best Practices

This guide distills a practical **analyze → optimize → verify** workflow for the modern OpenCL ecosystem, with an emphasis on OpenCL 3.0+, strict correctness, and cross-platform architectural differences.

## 1) Use a Three-Stage Optimization Loop

### Stage A — Understand and baseline first

- Define the kernel's objective and constraints: throughput, latency, memory footprint, and portability targets.
- Build a correctness baseline on CPU or with a known-good reference kernel.
- Measure before tuning: capture wall time, kernel time from OpenCL event profiling, and effective bandwidth.

### Stage B — Optimize with one hypothesis at a time

- Change one variable per iteration: work-group size, memory layout, vector width, use of local memory, or compiler flags.
- Keep changes small and attributable.
- Re-profile after every change.

### Stage C — Verify correctness and regressions

- Compare outputs against a reference within numerical tolerance.
- Add edge-case inputs: tiny tensors, odd dimensions, large dimensions, and non-multiples of the local size.
- Keep a small benchmark table by device and driver so regressions are obvious.

## 2) Correctness, Safety, and the Host API

### Prefer the C++ API (`opencl.hpp`) for modern codebases

If your host application is C++, use the official Khronos C++ wrapper. It manages resource lifetimes with RAII and can be configured to throw C++ exceptions on API errors, which removes a large amount of raw C API boilerplate.

```cpp
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

try {
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    // ... setup and enqueue ...
} catch (const cl::Error& e) {
    std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
    std::abort();
}
```

### Fallback: explicit error checking with the C API

If you are constrained to C, manually check every API return value and always retrieve build logs on compilation failures.

```cpp
#define CL_CHECK(call)                                                        \
  do {                                                                        \
    cl_int err__ = (call);                                                    \
    if (err__ != CL_SUCCESS) {                                                \
      fprintf(stderr, "OpenCL error %s:%d: %d\n", __FILE__, __LINE__, err__); \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
```

```cpp
inline void CL_CHECK_BUILD(cl_int err, cl_program program, cl_device_id device) {
  if (err == CL_SUCCESS) return;

  size_t log_size = 0;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

  if (log_size > 0) {
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          log_size, log.data(), nullptr);
    fprintf(stderr, "OpenCL build error: %d\n%s\n", (int)err, log.data());
  }

  std::abort();
}
```

### Validate device capabilities early

OpenCL 3.0 makes many OpenCL 2.x features optional.

- Never assume support for sub-groups, FP16, FP64, or SVM.
- Query capabilities with `clGetDeviceInfo` or `cl::Device::getInfo` during initialization.
- Gracefully fall back or stop early when required features are unavailable.

## 3) In-Kernel Feature Checks

When writing kernels for OpenCL C 3.0, use built-in feature macros such as `__opencl_c_*` to guard optional functionality inside the `.cl` source.

### Example: precision fallback

```c
#ifdef __opencl_c_fp16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half scalar_t;
#else
  typedef float scalar_t;
#endif
```

### Example: sub-group fallback

```c
#ifdef __opencl_c_subgroups
  sum = sub_group_reduce_add(val);
#else
  sum = local_memory_reduce(val, local_buffer);
#endif
```

## 4) Memory Best Practices

### Prefer coalesced global memory access

- Adjacent work-items should load adjacent elements whenever possible.
- Avoid stride-heavy patterns in the innermost access path.

### Use local memory for reuse, not automatically

Desktop and mobile OpenCL devices can behave very differently.

- On many desktop GPUs, `__local` maps to dedicated on-chip memory.
- On many mobile or embedded GPUs, `__local` may be backed by cache or global memory.

Rule of thumb: tile only when data reuse is high enough to offset barrier and local-memory overhead. On mobile targets, always compare a tiled `__local` kernel against a simple coalesced global-memory kernel.

## 5) Kernel Configuration and Occupancy

### Start with sane defaults

- Begin with `64-256` work-items per work-group for many scalar kernels.
- Query `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` and align the local size to it.
- Keep local size within both device and kernel limits from `clGetKernelWorkGroupInfo`.

### Tune for your bottleneck

- If memory-bound, improve access patterns and reduce bytes moved.
- If compute-bound, reduce instruction count, increase instruction-level parallelism, and use native math such as `native_exp` or `native_sin` when accuracy allows.
- If residency is low, inspect register pressure, private-memory spills, and local-memory usage.

## 6) Reduce Sub-Group and Wavefront Waste

- Minimize branch divergence in hot loops.
- Move invariant conditionals outside inner loops.
- Use sub-group operations, guarded by `__opencl_c_subgroups`, for reductions or scans when supported.
- Do not hard-code sub-group width assumptions. Query with `get_sub_group_size()` when needed.
- Prefer predication-friendly code paths.

## 7) Practical Profiling Workflow

- Use OpenCL event profiling or timeline tools to check enqueue overhead, launch gaps, host-device overlap, and queue concurrency.
- Use vendor profilers to inspect memory throughput, cache behavior, occupancy or residency, stall reasons, and instruction mix.
- Use Oclgrind and similar debugging tools to catch memory, race, and synchronization issues before and after major rewrites.

## 8) Common Optimization Playbook

### Leverage OpenCL JIT compilation

OpenCL compiles from source at runtime, which gives you a useful tuning mechanism.

- Pass compile-time constants through `clBuildProgram` options such as `-D TILE_SIZE=16 -D UNROLL_FACTOR=4`.
- Prefer compile-time constants over runtime kernel arguments when they enable loop unrolling, constant folding, and lower register usage.

### Know why you are vectorizing

- Use `float2`, `float4`, and similar vector types only when alignment is guaranteed.
- On many modern desktop GPUs, vector types mainly help memory efficiency.
- On some older mobile or embedded designs, vector types may also help compute utilization.

### General compute optimizations

- Fuse kernels when global-memory round trips dominate.
- Hoist repeated computations out of loops.
- Precompute constants and use `__constant` memory for broadcast-style reads.

## 9) Code Review Checklist

Before merging OpenCL changes, verify:

- [ ] Correctness validated against reference outputs.
- [ ] If using C++, `opencl.hpp` is used with exception handling. If using C, all API calls and builds are checked explicitly.
- [ ] Device capabilities for optional OpenCL 3.0 features are queried on the host.
- [ ] In-kernel fallback macros such as `__opencl_c_fp16` and `__opencl_c_subgroups` are used where needed.
- [ ] Memory access patterns are coalesced in critical loops.
- [ ] Work-group size was chosen from measured data, not guesswork.
- [ ] Register, private-memory, and local-memory usage was reviewed for cross-platform performance penalties.
- [ ] Profiling evidence includes before and after numbers.

## 10) Definition of Done for OpenCL Optimization

An OpenCL optimization is done when all of the following are true:

- It is measurably faster on representative workloads and target devices.
- It preserves numerical correctness within documented tolerance.
- It does not introduce portability or maintainability hazards such as hard-coded vendor assumptions, unverified extension usage, fragile barrier usage, or unexplained local sizes.
- It gracefully falls back or stops with a clear error on OpenCL 3.0 devices that lack required optional features.
- It includes enough benchmark context, including device, driver, build options, and problem size, for future regression checks.
