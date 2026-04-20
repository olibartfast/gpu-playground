# CLAUDE.md

> `AGENTS.md` at the repo root is the single source of truth for repository conventions, agent layout, and workflow guidance.
> Keep this file aligned with `AGENTS.md` and prefer updating `AGENTS.md` first when repo rules change.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A collection of standalone GPU kernel implementations (CUDA + OpenCL) for GPU programming experimentation. Each kernel directory under `source/` is self-contained with its own `CMakeLists.txt`. The project uses C++17/CUDA 17 with CMake 3.20+ and supports both CUDA and OpenCL backends via the `USE_OPENCL` CMake flag.

## Build Commands

```bash
# CUDA build (default: Tesla T4 / CC 7.0)
cmake --preset default
cmake --build --preset default -j$(nproc)

# Other presets: native (auto-detect GPU), ampere (SM 8.6), release (native + optimized)
cmake --preset native && cmake --build --preset native -j$(nproc)

# OpenCL build
cmake -B build/opencl -DUSE_OPENCL=ON
cmake --build build/opencl -j$(nproc)

# Build for a specific GPU architecture (e.g., Ampere/RTX 3080 = 86)
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
cmake --build . -j$(nproc)

# Build a single kernel target
cmake --build --preset default --target gemm

# Use "native" to auto-detect the local GPU (CMake 3.23+)
cmake -DCMAKE_CUDA_ARCHITECTURES=native ..

# Disable a specific kernel (feature flag)
cmake -DGPU_ENABLE_GEMM=OFF ..
```

Compiled binaries land in `build/<preset>/source/<kernel-name>/<kernel-name>` when using presets,
or `build/source/<kernel-name>/<kernel-name>` with a manual build.

## Running & Profiling

```bash
# Run a kernel directly
./build/default/source/gemm/gemm

# Profile with the provided analysis script
./cuda_perf_analysis.sh ./build/default/source/gemm/gemm
./cuda_perf_analysis.sh --skip-build ./build/default/source/softmax/softmax
```

The `cuda_perf_analysis.sh` script auto-detects `nvprof`, `ncu`, or `nsys` and prints a performance checklist.

## Project Structure

All source code lives under `source/`. Each kernel has a `cuda/` and `opencl/` subdirectory containing the backend-specific implementation, and a shared `main.cpp` test harness that selects the backend at compile time via `#ifdef GPU_OPENCL_BACKEND`.

```
source/
├── utils/
│   ├── cuda_helpers.h      ← CUDA_CHECK macro + getTime()
│   └── opencl_helpers.h    ← CL_CHECK macro + clSetupGPU/clBuildFromSource/clTeardown
├── gemm/
│   ├── CMakeLists.txt      ← if(USE_OPENCL) / else() block
│   ├── main.cpp            ← #ifdef GPU_OPENCL_BACKEND; calls only *_cpu()/*_gpu() wrappers
│   ├── cuda/
│   │   ├── gemm.h          ← CUDA declarations + host-pointer wrapper decl
│   │   └── gemm.cpp        ← CUDA kernel + host-pointer wrapper implementation
│   └── opencl/
│       ├── gemm.h          ← backend-agnostic API (no OpenCL types)
│       └── gemm.cpp        ← OpenCL kernel string + implementation
└── ...                     ← same layout for all 14 kernels
```

## Adding a New Kernel

See `docs/adding-a-new-kernel.md` for the full step-by-step guide including code templates for all four files.

Summary:
1. Create `source/<kernel-name>/cuda/<kernel>.{h,cpp}` and `opencl/<kernel>.{h,cpp}`.
2. Write `source/<kernel-name>/main.cpp` using `#ifdef GPU_OPENCL_BACKEND` and calling only `*_cpu()`/`*_gpu()` wrappers.
3. Add `source/<kernel-name>/CMakeLists.txt` with an `if(USE_OPENCL)/else()` block.
4. Register with `option(GPU_ENABLE_<NAME> ...)` + `add_subdirectory(...)` in the root `CMakeLists.txt`.

## CUDA Architecture Notes

- Default target: **SM 7.0** (Tesla T4) — set in root `CMakeLists.txt`.
- The `-real` suffix (e.g., `70-real`) produces SASS binary code instead of PTX, reducing JIT overhead at the cost of larger binaries.
- For CI/distribution builds specify an explicit list: `70-real;75-real;80-real;86`.
- Architecture-accelerated features on Hopper/Blackwell require the `a` suffix (e.g., `90a`).

## CUDA Programming Rules (from `docs/cuda-agent-guide.md`)

Key rules enforced in this codebase:

- **Thread block size** must be a multiple of 32; use 128–256 threads/block as a starting point.
- **Memory coalescing**: adjacent threads must access adjacent memory; prefer SoA over AoS.
- **Shared memory**: use for data accessed more than once; align data to 32-byte boundaries.
- **Error checking**: wrap all CUDA API calls with `CUDA_CHECK`; call `cudaGetLastError()` after every kernel launch. Wrap all OpenCL calls with `CL_CHECK`.
- **Avoid divergent branches** within a warp; prefer warp-aligned conditions.
- **Architecture guards**: use `#if __CUDA_ARCH__ >= 700` for Volta+ features.
- **No device-side allocation**: never call `malloc`/`new` inside kernels.
- **Fast math**: use `__sinf()`, `__expf()`, etc. when precision permits.

## Performance Profiling Targets

| Metric | Target |
|---|---|
| Global Load/Store Efficiency | >80% |
| Achieved Occupancy | >50% |
| Memory Throughput | >60% of theoretical |
| Warp Execution Efficiency | >90% |
