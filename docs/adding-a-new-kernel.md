# Adding a New Kernel

This guide walks through adding a self-contained CUDA kernel to the `source/` directory.

## Directory Layout

Every kernel follows the same three-file pattern:

```
source/<kernel-name>/
├── CMakeLists.txt
├── <kernel>.h       ← forward declarations
├── <kernel>.cpp     ← kernel + CPU reference implementations
└── main.cpp         ← test harness
```

> **File naming:** The directory name and binary name match. Source files inside use a short name that may differ (e.g. `interleave_arrays/` contains `interleave.h` / `interleave.cpp`).

---

## Step 1 — Create the source files

### `<kernel>.h`

Declare your `__global__` kernels, any GPU launch wrappers, and the CPU reference function:

```cpp
#pragma once
#include <cuda_runtime.h>

__global__ void myKernel(const float* input, float* output, int n);

void myKernelCpu(const float* input, float* output, int n);
```

### `<kernel>.cpp`

Implement everything declared in the header. Include `cuda_helpers.h` for `CUDA_CHECK` and `getTime()`:

```cpp
#include "my_kernel.h"
#include "cuda_helpers.h"

__global__ void myKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = input[idx] * 2.0f;
}

void myKernelCpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++)
        output[i] = input[i] * 2.0f;
}
```

Key rules (see `docs/cuda-copilot-rules.md` for the full list):
- Thread block size must be a multiple of 32; start with 128–256 threads/block.
- Wrap every CUDA API call with `CUDA_CHECK`.
- Call `cudaGetLastError()` after every kernel launch.
- No `malloc`/`new` inside kernels.

### `main.cpp`

Write the test harness: allocate host/device buffers, copy H→D, launch the kernel, copy D→H, validate against the CPU reference, and report timing:

```cpp
#include "my_kernel.h"
#include "cuda_helpers.h"
#include <cstdio>
#include <cstdlib>

int main() {
    const int N = 1 << 20;

    float* h_in  = new float[N];
    float* h_out_cpu = new float[N];
    float* h_out_gpu = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    // CPU reference
    double t0 = getTime();
    myKernelCpu(h_in, h_out_cpu, N);
    printf("CPU: %.3f ms\n", (getTime() - t0) * 1e3);

    // GPU
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));
    myKernel<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Validate
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out_cpu[i] - h_out_gpu[i]) > 1e-4f) errors++;
    printf(errors == 0 ? "PASSED\n" : "FAILED (%d errors)\n", errors);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    return errors != 0;
}
```

---

## Step 2 — Add `CMakeLists.txt`

```cmake
set_source_files_properties(my_kernel.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(my_kernel main.cpp my_kernel.cpp)
target_link_libraries(my_kernel PRIVATE cuda_helpers)
```

Source files use the `.cpp` extension and are compiled as CUDA via `set_source_files_properties`.

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
# Build everything (uses SM 7.0 by default)
cmake --preset default
cmake --build --preset default -j$(nproc)

# Or build just the new target
cmake --build --preset default --target my_kernel

# Run
./build/default/source/my_kernel/my_kernel

# Profile
./cuda_perf_analysis.sh ./build/default/source/my_kernel/my_kernel
```

---

## Checklist

- [ ] `<kernel>.h` — `#pragma once`, `cuda_runtime.h` include, `__global__` and CPU ref declarations
- [ ] `<kernel>.cpp` — implementations, `cuda_helpers.h` included, no device-side allocation
- [ ] `main.cpp` — allocate, H2D, launch, `cudaGetLastError()`, D2H, validate vs CPU, free
- [ ] `CMakeLists.txt` — `set_source_files_properties`, `add_executable`, link `cuda_helpers`
- [ ] Root `CMakeLists.txt` — `option(GPU_ENABLE_...)` + `add_subdirectory`
- [ ] Thread block size is a multiple of 32
- [ ] All CUDA API calls wrapped in `CUDA_CHECK`
