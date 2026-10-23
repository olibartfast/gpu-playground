# Adding a New Kernel

> Audience: humans, agents · Applies to: all · Last reviewed: 2026-10

This guide walks through adding a self-contained kernel to `source/`. A kernel
is one directory containing a backend-agnostic test harness, a CPU reference,
and one implementation per supported backend. CMake wires the variants together
behind a single `GPU_ENABLE_<KERNEL>` switch, so the same harness exercises
CUDA, HIP, OpenCL C, or OpenCL C++ without source changes.

The four-backend layout is the default. A task may explicitly add a CUDA-only
example when the operation or its dependency is backend-specific. In that case:

- keep CUDA code under `source/<kernel>/cuda/`
- declare `BACKENDS CUDA` so HIP and OpenCL configurations skip the target cleanly
- mark the backend coverage as CUDA-only in `Readme.md`
- validate an OpenCL configure and build, so the exception does not break the
  build matrix

## Directory Layout

Every portable kernel follows the same layout:

```
source/<kernel-name>/
├── CMakeLists.txt
├── cuda/
│   ├── <kernel>.h       ← CUDA declarations (includes cuda_runtime.h)
│   └── <kernel>.cpp     ← CUDA kernel + CPU reference implementations
├── hip/
│   ├── <kernel>.h       ← HIP declarations
│   └── <kernel>.cpp     ← HIP kernel + CPU reference implementations
├── opencl/
│   ├── <kernel>.h       ← OpenCL C API declarations (backend-agnostic API only)
│   └── <kernel>.cpp     ← OpenCL C API kernel + CPU reference implementations
├── opencl_cpp/
│   ├── <kernel>.h       ← OpenCL C++ wrapper declarations (backend-agnostic API only)
│   └── <kernel>.cpp     ← OpenCL C++ wrapper kernel + CPU reference implementations
└── main.cpp             ← backend-agnostic test harness
```

> **File naming:** The directory name and the binary name match. Source files
> inside may use a shorter name — for example, `interleave_arrays/` contains
> `interleave.h` and `interleave.cpp`.

HIP implementations use `hip/hip_runtime.h` and `hip_helpers.h`, with runtime
calls and launches checked using `HIP_CHECK`. Keep the same host-pointer wrapper
API. Add `GPU_HIP_BACKEND` to the harness selector and validate via
`cmake --preset hip`, `cmake --build --preset hip`, and `ctest --preset hip`.
See [the AMD neocloud runbook](../setup/amd-neocloud.md) for toolchain setup.

---

## Step 1 — Create the Source Files

Each backend exposes the same small, backend-agnostic API: a CPU reference and a
host-pointer GPU wrapper. Allocation, transfers, and cleanup stay inside the
implementation file. The harness sees only plain host pointers and scalar
arguments, which is what keeps it backend-agnostic.

### `cuda/<kernel>.h`

Declare the `__global__` kernels, the CPU reference, and the host-pointer GPU
wrapper used by `main.cpp`:

```cpp
#pragma once
#include <cuda_runtime.h>

__global__ void myKernel(const float* input, float* output, int n);

void myKernelCpu(const float* input, float* output, int n);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void myKernel_gpu(const float* h_input, float* h_output, int n);
```

### `cuda/<kernel>.cpp`

Implement the kernel, the reference, and the wrapper. Include `cuda_helpers.h`
for `CUDA_CHECK` and `getTime()`. The host-pointer wrapper owns all allocation,
transfer, and cleanup:

```cpp
#include "my_kernel.h"
#include "cuda_helpers.h"

__global__ void myKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = input[idx] * 2.0f;
}

void myKernelCpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = input[i] * 2.0f;
}

void myKernel_gpu(const float* h_input, float* h_output, int n) {
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_input, n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256, blocks = (n + threads - 1) / threads;
    myKernel<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_output, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}
```

The core CUDA rules apply here; the full list is in
[the CUDA optimization guide](../optimization/cuda.md).

- The thread block size must be a multiple of 32; start with 128–256 threads.
- Wrap every CUDA API call with `CUDA_CHECK`.
- Call `cudaGetLastError()` after every kernel launch.
- Never allocate memory inside a kernel.

### `opencl/<kernel>.h`

Expose only the backend-agnostic API — no OpenCL types in the header:

```cpp
#pragma once

void myKernelCpu(const float* input, float* output, int n);
void myKernel_gpu(const float* h_input, float* h_output, int n);
```

### `opencl/<kernel>.cpp`

Embed the kernel source as a string literal and use `opencl_c_helpers.h` for the
context, program, and teardown boilerplate:

```cpp
#include "my_kernel.h"
#include "opencl_c_helpers.h"
#include <cmath>

static const char* KERNEL_SOURCE = R"(
__kernel void myKernel(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx < n) output[idx] = input[idx] * 2.0f;
}
)";

void myKernelCpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = input[i] * 2.0f;
}

void myKernel_gpu(const float* h_input, float* h_output, int n) {
    cl_context ctx; cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * n, (void*)h_input, &err); CL_CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "myKernel", &err); CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &n));

    size_t local = clPreferredLocalSize(kernel, dev);
    size_t global = ((size_t)n + local - 1) / local * local;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, sizeof(float) * n, h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_in); clReleaseMemObject(d_out);
    clTeardown(ctx, queue, prog, kernel);
}
```

### `opencl_cpp/<kernel>.h`

Use the same backend-agnostic API as the OpenCL C header — again, no OpenCL
types:

```cpp
#pragma once

void myKernelCpu(const float* input, float* output, int n);
void myKernel_gpu(const float* h_input, float* h_output, int n);
```

### `opencl_cpp/<kernel>.cpp`

Use `opencl_helpers.h` (the C++ wrapper helper) with RAII `cl::` objects and
exception-based error handling:

```cpp
#include "my_kernel.h"
#include "opencl_helpers.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
__kernel void myKernel(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0);
    if (idx < n) output[idx] = input[idx] * 2.0f;
}
)";

void myKernelCpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = input[i] * 2.0f;
}

void myKernel_gpu(const float* h_input, float* h_output, int n) {
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_in(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * n, const_cast<float*>(h_input));
        cl::Buffer d_out(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * n);

        cl::Kernel kernel(prog, "myKernel");
        kernel.setArg(0, d_in);
        kernel.setArg(1, d_out);
        kernel.setArg(2, n);

        size_t local = clppPreferredLocalSize(kernel, dev);
        size_t global = ((size_t)n + local - 1) / local * local;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(global), cl::NDRange(local));
        queue.finish();
        queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(float) * n, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
```

Special cases:

- **Local (shared) memory:** pass `cl::Local(sharedSize)` as a kernel argument
  instead of a buffer.
- **2D dispatch** for matrix kernels: use `cl::NDRange(cols, rows)` for the
  global and local ranges.
- **Build options:** pass extra defines through
  `clppBuildProgram(ctx, dev, src, "-D TILE_SIZE=16")`.

### `main.cpp`

The harness selects a backend at compile time and calls only the
backend-agnostic API:

```cpp
#if defined(GPU_HIP_BACKEND)
#include "hip/my_kernel.h"
#elif defined(GPU_OPENCL_CPP_BACKEND)
#include "opencl_cpp/my_kernel.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/my_kernel.h"
#else
#include "cuda/my_kernel.h"
#endif
#include <cstdio>
#include <cmath>
#include <vector>
#include "benchmark.h"   // from source/utils/, available to all backends

int main() {
    const int N = 1 << 20;
    std::vector<float> h_in(N), h_out_cpu(N), h_out_gpu(N);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    // Never time a single call: the first invocation pays CUDA context creation
    // or OpenCL program build, which can dwarf the kernel itself. benchmark()
    // discards warm-up iterations and reports a distribution over the rest.
    BenchResult cpu_bench = benchmark([&] { myKernelCpu(h_in.data(), h_out_cpu.data(), N); });
    BenchResult gpu_bench = benchmark([&] { myKernel_gpu(h_in.data(), h_out_gpu.data(), N); });
    printBench("CPU:", cpu_bench);
    printBench("GPU:", gpu_bench);
    printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out_cpu[i] - h_out_gpu[i]) > 1e-4f) errors++;
    printf(errors == 0 ? "PASSED\n" : "FAILED (%d errors)\n", errors);
    return errors != 0;
}
```

Three timing paths are available:

- **Ordinary kernels:** `benchmark(fn)`.
- **In-place kernels, or GEMM's `beta * C` term:** `benchmarkWithReset(fn,
  reset)`. `reset` restores the buffer before every call and stays outside the
  timed region. Without it, each iteration measures different work and the
  correctness check compares against an accumulated result rather than one clean
  pass.
- **Kernel-only time:** if the GPU wrapper exposes a kernel time through a
  `float* kernel_time_ms` out-parameter, use `benchmarkDevice(fn)` or
  `benchmarkDeviceWithReset(fn, reset)`. `fn` returns that call's
  device-measured milliseconds, and the result's `end_to_end` and `device`
  fields are each a `BenchResult`, so the harness can report kernel-only time
  alongside host overhead.

---

## Step 2 — Add `CMakeLists.txt`

The shared helper derives the backend source path from the kernel name and
registers the target with CTest:

```cmake
gpu_add_kernel(my_kernel)
```

This replaces a hand-written backend selection block. It selects the requested
backend, links the matching helper library, defines `GPU_OPENCL_BACKEND` /
`GPU_OPENCL_CPP_BACKEND` when needed, and skips the target with a status message
if that backend is not implemented.

Source files keep the `.cpp` extension; CUDA files are compiled as CUDA through
`set_source_files_properties`.

If the source files use a shorter name than the directory, pass it explicitly:

```cmake
gpu_add_kernel(interleave_arrays SOURCE_NAME interleave)
```

Additional backend sources, definitions, and compile options can be supplied
with `EXTRA_SOURCES`, `COMPILE_DEFINITIONS`, and `COMPILE_OPTIONS`. Use
`NO_TEST` for a harness that needs arguments or user-supplied input.

---

## Step 3 — Register in the Root `CMakeLists.txt`

Add the kernel name to `GPU_KERNELS` in the root `CMakeLists.txt`:

```cmake
set(GPU_KERNELS
    ...
    my_kernel
)
```

The loop creates the `GPU_ENABLE_MY_KERNEL` option automatically and adds the
subdirectory when it is enabled. The flag lets users disable a kernel at
configure time:

```bash
cmake -DGPU_ENABLE_MY_KERNEL=OFF ..
```

---

## Step 4 — Update `Readme.md` (Required)

Every new kernel must update the root `Readme.md` in the same change:

- the project-structure kernel count or wording
- the kernel inventory table, including backend coverage
- any prerequisites or non-default build and run commands
- links to a kernel-specific README when setup takes more than one command

Agents must not present a new kernel as complete until this documentation is
accurate.

---

## Step 5 — Build and Run

```bash
# CUDA build (default: SM 7.5, RelWithDebInfo)
cmake --preset default
cmake --build --preset default -j$(nproc)

# Or build just the new target
cmake --build --preset default --target my_kernel

# Run
./build/default/source/my_kernel/my_kernel

# Run the registered regression test
ctest --preset default -R '^my_kernel$' --output-on-failure

# Profile
./cuda_perf_analysis.sh ./build/default/source/my_kernel/my_kernel

# OpenCL C API build
cmake -B build/opencl -DUSE_OPENCL=ON
cmake --build build/opencl -j$(nproc) --target my_kernel
./build/opencl/source/my_kernel/my_kernel

# OpenCL C++ wrapper build (requires: sudo apt install opencl-clhpp-headers)
cmake -B build/opencl_cpp -DUSE_OPENCL_CPP=ON
cmake --build build/opencl_cpp -j$(nproc) --target my_kernel
./build/opencl_cpp/source/my_kernel/my_kernel
```

---

## Optional: Python DSL Backend

A kernel may also — or, for a Python-only kernel such as `vector_addition`,
instead — ship a Python DSL backend at
`source/<kernel>/python/<dsl>/<kernel>.py`, where `<dsl>` is `triton` or
`cute-dsl`. These are standalone scripts run from the repository root
(`python source/<kernel>/python/<dsl>/<kernel>.py [--performance]`), not CMake
targets.

Follow the shared `gpu_bench` contract: call `require(...)` before importing the
DSL module so a missing dependency exits `77` (SKIPPED) rather than failing, and
validate and time `solve(...)` through `gpu_bench`. See "Testing And
Benchmarking" in [AGENTS.md](../../AGENTS.md) for the full contract.

---

## Checklist

- [ ] `cuda/<kernel>.h` — `#pragma once`, `cuda_runtime.h`, `__global__`
      declarations, host-pointer wrapper declaration
- [ ] `cuda/<kernel>.cpp` — CUDA kernel, CPU reference, host-pointer wrapper;
      `CUDA_CHECK` on every API call; `cudaGetLastError()` after launch
- [ ] `opencl/<kernel>.h` — `#pragma once`, backend-agnostic signatures only
      (no OpenCL types)
- [ ] `opencl/<kernel>.cpp` — kernel string literal, `opencl_c_helpers.h`,
      `CL_CHECK` on every API call, `clTeardown` to release resources
- [ ] `opencl_cpp/<kernel>.h` — `#pragma once`, same signatures as
      `opencl/<kernel>.h`
- [ ] `opencl_cpp/<kernel>.cpp` — kernel string literal, `opencl_helpers.h`,
      `cl::` RAII objects, `try/catch (cl::Error)`
- [ ] `main.cpp` — backend selector including `GPU_HIP_BACKEND`, `GPU_OPENCL_CPP_BACKEND` /
      `#elif GPU_OPENCL_BACKEND` / `#else` guard; calls only `*_cpu()` and
      `*_gpu()` wrappers; validates CPU against GPU output
- [ ] `main.cpp` — timings come from `benchmark()` / `benchmarkWithReset()` (or
      `benchmarkDevice()` / `benchmarkDeviceWithReset()` for kernel-only time)
      in `source/utils/benchmark.h`, never a bare `steady_clock` pair
- [ ] `CMakeLists.txt` — `gpu_add_kernel(<name>)`, with `SOURCE_NAME` /
      `BACKENDS` / `EXTRA_SOURCES` only when the defaults do not fit
- [ ] Root `CMakeLists.txt` — kernel name added to `GPU_KERNELS`
- [ ] CTest — `ctest --preset default -R '^<name>$'` passes
- [ ] Root `Readme.md` — kernel inventory, backend coverage, and required
      build/run instructions
- [ ] Thread block size is a multiple of 32
- [ ] All CUDA API calls use `CUDA_CHECK`; all OpenCL C calls use `CL_CHECK`;
      OpenCL C++ uses exception handling
- [ ] (optional) `python/<dsl>/<kernel>.py` — follows the `gpu_bench` contract
      in `AGENTS.md` and exits `0`, `1`, or `77`
- [ ] (optional) Root `Readme.md` — Python DSL backend noted in the kernel
      table and the Python DSL section
