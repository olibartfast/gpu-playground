# Repository Guidelines

## Project Structure & Module Organization
`source/` contains all kernel implementations. Each kernel lives in `source/<kernel>/` with a `main.cpp` harness plus backend-specific code under `cuda/`, `opencl/`, and `opencl_cpp/`. Shared helpers live in `source/utils/`. Top-level build configuration is in `CMakeLists.txt` and `CMakePresets.json`. Reference material and how-to guides are under `docs/`, and `.devcontainer/` holds the recommended CUDA development environment.

## Build, Test, and Development Commands
Use CMake presets for normal CUDA development:

```bash
cmake --preset default
cmake --build --preset default -j$(nproc)
```

Use `cmake --preset native` to target the local GPU, or `cmake --preset release` for optimized builds. For OpenCL backends, configure explicitly: `cmake -B build/opencl -DUSE_OPENCL=ON` or `cmake -B build/opencl_cpp -DUSE_OPENCL_CPP=ON`. Build a single kernel with `cmake --build build/default --target gemm`. Run the resulting harness from `build/<preset>/source/<kernel>/<kernel>`.

## Coding Style & Naming Conventions
Follow the existing C++17/CUDA style: 4-space indentation, braces on the same line, and small helper functions in `main.cpp` for validation and printing. Keep directory, target, and feature-flag names aligned: `source/value_clipping`, target `value_clipping`, flag `GPU_ENABLE_VALUE_CLIPPING`. Backend headers expose backend-agnostic APIs; backend-specific types stay in `.cpp` files. Reuse `CUDA_CHECK`, OpenCL helpers, and the compile-time backend switch pattern already used in `source/gemm/main.cpp`.

## Testing Guidelines
There is no separate unit-test framework; each kernel executable is its own test harness. Add at least one small deterministic case and one larger randomized case, comparing CPU and GPU/OpenCL results with a stated tolerance. Time every path with `benchmark()` from `source/utils/benchmark.h` (warm-up iterations then repeated timed runs, reported as a median plus spread) rather than a single `steady_clock` pair; use `benchmarkWithReset()` when the output buffer is also an input. Validate new kernels by running the relevant binary, for example `build/default/source/softmax/softmax`. If backend behavior differs, test the CUDA and OpenCL variants separately.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Add OpenCL C++ wrapper support`, `Refactor OpenCL kernels...`, and `Update README...`. Keep commit titles concise, capitalized, and action-first. PRs should describe the affected kernel or backend, list the build/test commands run, and note GPU architecture or runtime used. Include benchmark output or screenshots only when they clarify a performance or documentation change.

## Configuration Notes
Default CUDA builds target SM70 unless overridden. Use `CMAKE_CUDA_ARCHITECTURES` or the `native`/`ampere` presets when testing on different hardware, and disable unfinished kernels with `-DGPU_ENABLE_<KERNEL>=OFF`.
