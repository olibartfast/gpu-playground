# Copilot Instructions

> **Agents: follow and maintain [AGENTS.md](../AGENTS.md) at the repo root.**
> It is the single source of truth for repository conventions, agent layout, and workflow guidance.

## Project Overview

A collection of standalone GPU kernel implementations in C++17/CUDA 17 with OpenCL support, managed via CMake 3.20+. Each kernel under `source/` is a self-contained mini-project with CUDA and OpenCL backends and its own `CMakeLists.txt`. The default GPU target is **Tesla T4 (SM 7.0)**.

## Build Commands

```bash
# Configure + build CUDA (default: SM 7.0, Debug)
cmake --preset default
cmake --build --preset default -j$(nproc)

# Other presets: native (auto-detect GPU), ampere (SM 8.6), release (native + optimized)
cmake --preset native && cmake --build --preset native -j$(nproc)

# Build OpenCL C API backend
cmake -B build/opencl -DUSE_OPENCL=ON
cmake --build build/opencl -j$(nproc)

# Build OpenCL C++ wrapper backend
cmake -B build/opencl_cpp -DUSE_OPENCL_CPP=ON
cmake --build build/opencl_cpp -j$(nproc)

# Build a single kernel
cmake --build --preset default --target gemm

# Disable a kernel at configure time
cmake -DGPU_ENABLE_GEMM=OFF ..
```

Binary locations:
- CUDA presets: `build/<preset>/source/<kernel>/<kernel>`
- CUDA manual: `build/source/<kernel>/<kernel>`
- OpenCL C API: `build/opencl/source/<kernel>/<kernel>`
- OpenCL C++ wrapper: `build/opencl_cpp/source/<kernel>/<kernel>`

## Running and Profiling

```bash
# Run a kernel
./build/default/source/gemm/gemm

# Profile (auto-detects nvprof / ncu / nsys)
./cuda_perf_analysis.sh ./build/default/source/gemm/gemm
./cuda_perf_analysis.sh --skip-build ./build/default/source/softmax/softmax
```

## Architecture

Every kernel follows an identical layout with three backend subdirectories and a shared `main.cpp`:

```
source/<kernel-name>/
├── CMakeLists.txt          ← if(USE_OPENCL_CPP)/elseif(USE_OPENCL)/else() block
├── main.cpp                ← three-way backend guard; calls only *_cpu()/*_gpu()
├── cuda/
│   ├── <kernel>.h          ← #pragma once, cuda_runtime.h, __global__ decls, host-pointer wrapper decl
│   └── <kernel>.cpp        ← CUDA kernel + CPU ref + host-pointer wrapper
├── opencl/
│   ├── <kernel>.h          ← backend-agnostic API only (no OpenCL types)
│   └── <kernel>.cpp        ← kernel as C string literal + OpenCL C API boilerplate
└── opencl_cpp/
    ├── <kernel>.h          ← backend-agnostic API only (no OpenCL types)
    └── <kernel>.cpp        ← kernel as C string literal + OpenCL C++ wrapper (cl:: RAII objects)
```

The `main.cpp` backend guard is three-way:
```cpp
#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/<kernel>.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/<kernel>.h"
#else
#include "cuda/<kernel>.h"
#endif
```

The shared utility libraries live in `source/utils/`:
- `cuda_helpers.h` — `CUDA_CHECK(call)`, `getTime()`
- `opencl_c_helpers.h` — `CL_CHECK(call)`, OpenCL C API helpers (`clSetupGPU`, `clBuildFromSource`, `clTeardown`)
- `opencl_helpers.h` — OpenCL C++ wrapper helpers using `cl::` RAII objects and exception-based error handling

> **File naming:** The directory name and resulting binary name match. Source files inside may use a shorter name (e.g., `interleave_arrays/` contains `interleave.h` / `interleave.cpp`).

Source files use the `.cpp` extension but are compiled as CUDA via `set_source_files_properties`.

## Adding a New Kernel

1. Create `source/<kernel-name>/cuda/<kernel>.{h,cpp}` and `opencl/<kernel>.{h,cpp}`.
2. Write `source/<kernel-name>/main.cpp` with `#ifdef GPU_OPENCL_BACKEND` guard; call only `*_cpu()`/`*_gpu()` wrappers.
3. Add `source/<kernel-name>/CMakeLists.txt` with an `if(USE_OPENCL)/else()` block.
4. Register in root `CMakeLists.txt`:
   ```cmake
   option(GPU_ENABLE_<KERNEL_UPPER> "Build <kernel> kernel" ON)
   if(GPU_ENABLE_<KERNEL_UPPER>)
       add_subdirectory(source/<kernel-name>)
   endif()
   ```

See `docs/adding-a-new-kernel.md` for full code templates for all four files.

## CUDA Coding Rules

These rules apply to all kernels in this repo:

- **Thread block size** must be a multiple of 32; use 128–256 as a starting point.
- **Memory coalescing**: adjacent threads must access adjacent memory. Prefer SoA over AoS.
- **Shared memory**: use for data accessed more than once; align to 32-byte boundaries.
- **Error checking**: wrap every CUDA API call with `CUDA_CHECK`; every OpenCL call with `CL_CHECK`; call `cudaGetLastError()` after every CUDA kernel launch.
- **No device-side allocation**: never call `malloc`/`new` inside kernels.
- **Avoid warp divergence**: prefer warp-aligned conditions over thread-divergent branches.
- **Architecture guards**: use `#if __CUDA_ARCH__ >= 700` for Volta+ features (warp shuffle, tensor cores).
- **Fast math**: use `__sinf()`, `__expf()`, etc. when precision permits.

## Test Harness Pattern (`main.cpp`)

Every `main.cpp` uses this pattern:
1. `#ifdef GPU_OPENCL_BACKEND` / `#else` to include the right backend header.
2. Call `*_cpu()` for the CPU reference and time with `std::chrono`.
3. Call the host-pointer `*_gpu()` wrapper (all allocation/transfer/launch is encapsulated inside).
4. Compare outputs against the CPU reference with tolerance `1e-4f` to `1e-5f`.
5. Return non-zero on validation failure.

All CUDA-specific plumbing (`cudaMalloc`, `cudaMemcpy`, `<<<>>>`, `cudaEvent_t`) lives exclusively inside `cuda/<kernel>.cpp`; all OpenCL plumbing (`clCreateBuffer`, `clEnqueueNDRangeKernel`, etc.) lives exclusively inside `opencl/<kernel>.cpp`.

## Performance Targets

| Metric | Target |
|---|---|
| Global Load/Store Efficiency | >80% |
| Achieved Occupancy | >50% |
| Memory Throughput | >60% of theoretical |
| Warp Execution Efficiency | >90% |

## CUDA Architecture Notes

- The `-real` suffix (e.g., `70-real`) emits SASS instead of PTX — reduces JIT overhead, increases binary size.
- For CI/distribution, specify an explicit list: `70-real;75-real;80-real;86`.
- Hopper/Blackwell architecture-accelerated features require the `a` suffix (e.g., `90a`).
- Extended shared memory (up to 164KB/block) is available on CC 8.0+ with explicit opt-in.
