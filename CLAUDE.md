# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### CUDA (default)
```bash
# Configure and build all kernels (targets Tesla T4 / SM70 by default)
cmake --preset default
cmake --build --preset default -j$(nproc)

# Target the local GPU (requires CMake 3.23+)
cmake --preset native && cmake --build --preset native -j$(nproc)

# Optimized release build
cmake --preset release && cmake --build --preset release -j$(nproc)
```

### Build a single kernel
```bash
cmake --build build/default --target gemm
```

### OpenCL backends
```bash
# OpenCL C API
cmake -B build/opencl -DUSE_OPENCL=ON
cmake --build build/opencl -j$(nproc)

# OpenCL C++ wrapper (requires: sudo apt install opencl-clhpp-headers)
cmake -B build/opencl_cpp -DUSE_OPENCL_CPP=ON
cmake --build build/opencl_cpp -j$(nproc)
```

### Disable a specific kernel at configure time
```bash
cmake -DGPU_ENABLE_GEMM=OFF --preset default
```

## Running Kernels

Each kernel binary is its own test harness (no separate test framework):
```bash
# CUDA build
./build/default/source/<kernel>/<kernel>
# e.g.
./build/default/source/gemm/gemm

# OpenCL builds
./build/opencl/source/<kernel>/<kernel>
./build/opencl_cpp/source/<kernel>/<kernel>
```

## Profiling
```bash
./cuda_perf_analysis.sh ./build/default/source/gemm/gemm
```
Auto-detects nvprof / ncu / nsys.

## Architecture

### Three-backend design
Every kernel under `source/<kernel>/` supports three backends selected at compile time:
- `cuda/` — CUDA implementation (`.cpp` files compiled as CUDA via `set_source_files_properties`)
- `opencl/` — OpenCL C API implementation
- `opencl_cpp/` — OpenCL C++ wrapper (RAII `cl::` objects, exception-based error handling)

`main.cpp` is the **backend-agnostic test harness** that selects the backend via a 3-way `#ifdef`:
```cpp
#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/kernel.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/kernel.h"
#else
#include "cuda/kernel.h"
#endif
```

### Shared utilities (`source/utils/`)
- `benchmark.h` — backend-agnostic timing: `benchmark()` / `benchmarkWithReset()` (warm-up + repeated iterations), `printBench`, `printSpeedup`
- `cuda_helpers.h` — `CUDA_CHECK` macro + `getTime()`
- `opencl_c_helpers.h` — `CL_CHECK`, `clSetupGPU`, `clBuildFromSource`, `clTeardown`
- `opencl_helpers.h` — `clppGetGPUDevice`, `clppBuildProgram`, `clppPreferredLocalSize`

### Host-pointer wrapper pattern
Each backend exposes a backend-agnostic API to `main.cpp`:
- `<kernel>_gpu(const float* h_input, float* h_output, int n)` — handles all allocation, H2D/D2H transfers, and cleanup internally
- `<kernel>Cpu(...)` — CPU reference implementation for validation

Backend-specific types (CUDA pointers, `cl_mem`, `cl::Buffer`) never appear in headers.

### CMake structure
- Root `CMakeLists.txt` selects backend (CUDA vs OpenCL) and has one `GPU_ENABLE_<KERNEL>` option + `add_subdirectory` per kernel
- Per-kernel `CMakeLists.txt` uses `if(USE_OPENCL_CPP) / elseif(USE_OPENCL) / else()` to compile the right sources and link the right helper library (`cuda_helpers`, `opencl_helpers`, or `opencl_cpp_helpers`)
- Default CUDA architecture: SM70 (Tesla T4); override with `CMAKE_CUDA_ARCHITECTURES` or use `native`/`ampere` presets

## Adding a New Kernel

See `docs/adding-a-new-kernel.md` for the full walkthrough. Summary:

1. Create `source/<kernel>/` with `cuda/`, `opencl/`, `opencl_cpp/` subdirs and `main.cpp`
2. Add per-kernel `CMakeLists.txt` with the 3-way backend block
3. Register in root `CMakeLists.txt`: add `option(GPU_ENABLE_<KERNEL> ...)` + `add_subdirectory`

**Checklist for kernel implementations:**
- Thread block size must be a multiple of 32; start with 128–256 threads/block
- Wrap every CUDA API call with `CUDA_CHECK`; call `cudaGetLastError()` after every kernel launch
- Wrap every OpenCL C call with `CL_CHECK`; use `try/catch (cl::Error)` in the C++ wrapper path
- OpenCL kernels embedded as string literals; use `clPreferredLocalSize` / `clppPreferredLocalSize` for work-group sizing
- `main.cpp` validates GPU output against CPU reference (tolerance ~1e-4f)
- `main.cpp` times via `benchmark()` from `utils/benchmark.h`, never a single `steady_clock` pair — the first call pays CUDA context creation / OpenCL program build. Use `benchmarkWithReset()` when the output buffer is also an input (in-place kernels, GEMM's `beta * C`)

## Coding Conventions
- C++17 throughout; 4-space indentation, braces on same line
- Directory name = target name = binary name (e.g. `source/value_clipping` → target `value_clipping` → flag `GPU_ENABLE_VALUE_CLIPPING`)
- CUDA source files use `.cpp` extension and are tagged as CUDA via `set_source_files_properties`
- Commit messages: short imperative subjects, capitalized, action-first (e.g. `Add softmax OpenCL backend`)
