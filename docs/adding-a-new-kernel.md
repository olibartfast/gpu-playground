# Adding a New Kernel

This guide walks through adding a self-contained CUDA + OpenCL kernel to the `source/` directory.

## Directory Layout

Every kernel follows the same layout with separate `cuda/` and `opencl/` subdirectories:

```
source/<kernel-name>/
├── CMakeLists.txt
├── cuda/
│   ├── <kernel>.h       ← CUDA declarations (includes cuda_runtime.h)
│   └── <kernel>.cpp     ← CUDA kernel + CPU reference implementations
├── opencl/
│   ├── <kernel>.h       ← OpenCL declarations (backend-agnostic API only)
│   └── <kernel>.cpp     ← OpenCL kernel + CPU reference implementations
└── main.cpp             ← backend-agnostic test harness
```

> **File naming:** The directory name and binary name match. Source files inside use a short name that may differ (e.g. `interleave_arrays/` contains `interleave.h` / `interleave.cpp`).

---

## Step 1 — Create the source files

### `cuda/<kernel>.h`

Declare your `__global__` kernels, device-pointer launch wrappers, the CPU reference, and a host-pointer GPU wrapper used by `main.cpp`:

```cpp
#pragma once
#include <cuda_runtime.h>

__global__ void myKernel(const float* input, float* output, int n);

void myKernelCpu(const float* input, float* output, int n);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void myKernel_gpu(const float* h_input, float* h_output, int n);
```

### `cuda/<kernel>.cpp`

Implement everything. Include `cuda_helpers.h` for `CUDA_CHECK` and `getTime()`. The host-pointer wrapper handles all allocation, transfer, and cleanup:

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

Key rules (see `docs/agentic_ai/cuda-copilot-rules.md` for the full list):
- Thread block size must be a multiple of 32; start with 128–256 threads/block.
- Wrap every CUDA API call with `CUDA_CHECK`.
- Call `cudaGetLastError()` after every kernel launch.
- No `malloc`/`new` inside kernels.

### `opencl/<kernel>.h`

Expose only the backend-agnostic API — no OpenCL types in the header:

```cpp
#pragma once

void myKernelCpu(const float* input, float* output, int n);
void myKernel_gpu(const float* h_input, float* h_output, int n);
```

### `opencl/<kernel>.cpp`

Embed the kernel source as a string literal and use `opencl_helpers.h` for boilerplate:

```cpp
#include "my_kernel.h"
#include "opencl_helpers.h"
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

    size_t local = 256, global = ((size_t)n + 255) / 256 * 256;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, sizeof(float) * n, h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_in); clReleaseMemObject(d_out);
    clTeardown(ctx, queue, prog, kernel);
}
```

### `main.cpp`

The test harness selects the backend at compile time and only calls the backend-agnostic API:

```cpp
#ifdef GPU_OPENCL_BACKEND
#include "opencl/my_kernel.h"
#else
#include "cuda/my_kernel.h"
#endif
#include <cstdio>
#include <chrono>
#include <cmath>
#include <vector>

int main() {
    const int N = 1 << 20;
    std::vector<float> h_in(N), h_out_cpu(N), h_out_gpu(N);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    auto t0 = std::chrono::steady_clock::now();
    myKernelCpu(h_in.data(), h_out_cpu.data(), N);
    auto t1 = std::chrono::steady_clock::now();
    printf("CPU: %.3f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    auto g0 = std::chrono::steady_clock::now();
    myKernel_gpu(h_in.data(), h_out_gpu.data(), N);
    auto g1 = std::chrono::steady_clock::now();
    printf("GPU: %.3f ms\n", std::chrono::duration<double, std::milli>(g1 - g0).count());

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out_cpu[i] - h_out_gpu[i]) > 1e-4f) errors++;
    printf(errors == 0 ? "PASSED\n" : "FAILED (%d errors)\n", errors);
    return errors != 0;
}
```

---

## Step 2 — Add `CMakeLists.txt`

```cmake
if(USE_OPENCL)
    add_executable(my_kernel main.cpp opencl/my_kernel.cpp)
    target_link_libraries(my_kernel PRIVATE opencl_helpers)
    target_compile_definitions(my_kernel PRIVATE GPU_OPENCL_BACKEND)
else()
    set_source_files_properties(cuda/my_kernel.cpp main.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(my_kernel main.cpp cuda/my_kernel.cpp)
    target_link_libraries(my_kernel PRIVATE cuda_helpers)
endif()
```

Source files use the `.cpp` extension. CUDA files are compiled as CUDA via `set_source_files_properties`.

---

## Step 3 — Register in the root `CMakeLists.txt`

Add an `option` and an `add_subdirectory` call alongside the existing kernels:

```cmake
option(GPU_ENABLE_MY_KERNEL "Build my_kernel kernel" ON)

if(GPU_ENABLE_MY_KERNEL)
    add_subdirectory(source/my_kernel)
endif()
```

The `GPU_ENABLE_<NAME>` flag lets users selectively disable a kernel at configure time:

```bash
cmake -DGPU_ENABLE_MY_KERNEL=OFF ..
```

---

## Step 4 — Build and run

```bash
# CUDA build (default: SM 7.0)
cmake --preset default
cmake --build --preset default -j$(nproc)

# Or build just the new target
cmake --build --preset default --target my_kernel

# Run
./build/default/source/my_kernel/my_kernel

# Profile
./cuda_perf_analysis.sh ./build/default/source/my_kernel/my_kernel

# OpenCL build
cmake -B build/opencl -DUSE_OPENCL=ON
cmake --build build/opencl -j$(nproc) --target my_kernel
./build/opencl/source/my_kernel/my_kernel
```

---

## Checklist

- [ ] `cuda/<kernel>.h` — `#pragma once`, `cuda_runtime.h`, `__global__` decls, host-pointer wrapper decl
- [ ] `cuda/<kernel>.cpp` — CUDA kernel + CPU ref + host-pointer wrapper; `CUDA_CHECK` on all API calls; `cudaGetLastError()` after launch
- [ ] `opencl/<kernel>.h` — `#pragma once`, backend-agnostic function signatures only (no OpenCL types)
- [ ] `opencl/<kernel>.cpp` — kernel string literal, `CL_CHECK` on all API calls, `clTeardown` to release resources
- [ ] `main.cpp` — `#ifdef GPU_OPENCL_BACKEND` guard; calls only `*_cpu()` and `*_gpu()` wrappers; validates CPU vs GPU output
- [ ] `CMakeLists.txt` — `if(USE_OPENCL)` / `else()` block; correct `set_source_files_properties` for CUDA path
- [ ] Root `CMakeLists.txt` — `option(GPU_ENABLE_...)` + `add_subdirectory`
- [ ] Thread block size is a multiple of 32
- [ ] All CUDA API calls wrapped in `CUDA_CHECK`; all OpenCL calls wrapped in `CL_CHECK`

