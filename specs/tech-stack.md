# Technical Boundaries

Evidence: [CMake](../CMakeLists.txt) and the [repository
contract](../AGENTS.md).

- The codebase is C++17, CUDA, OpenCL C, and OpenCL C++ wrappers, built with
  CMake.
- Implementations stay in their existing backend directories, behind shared
  backend-agnostic harnesses.
- Each executable supplies its own correctness harness; backend availability
  varies by kernel.
- Use `benchmark()` or `benchmarkWithReset()` for repeated timing, with warm-up,
  medians, and spread. Use `benchmarkDevice()` when a kernel-only timing
  boundary is needed alongside end-to-end time. Every reported measurement
  identifies the device, precision, shape, and timing boundary.
- Optimization work follows the existing [CUDA](../docs/optimization/cuda.md)
  and [OpenCL](../docs/optimization/opencl.md) guides.
- Optional Python DSL backends (Triton and CuTe DSL) live at
  `source/<kernel>/python/<dsl>/<kernel>.py`. They are standalone scripts, not
  CMake targets, and they share a torch-based benchmarking helper,
  `source/utils/python/gpu_bench.py`, rather than `source/utils/benchmark.h`.
- AMD GPU support is planned through HIP for the MI300X (`gfx942`), specified in
  [the HIP backend packet](2026-10-19-hip-backend/requirements.md). HIP is not
  implemented yet; CUDA remains the default, and CUDA and OpenCL remain
  supported.

- GLSL compute backends use OpenGL 4.3/EGL or Vulkan with C++17 host code.
  Shared shader sources live in `glsl/`; API host implementations live in
  `opengl/` and `vulkan/`. Initial coverage is sigmoid; see
  [GLSL backend packet](2026-10-24-glsl-backends/requirements.md).
