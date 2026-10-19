# Technical boundaries

Evidence: [CMake](../CMakeLists.txt) and [repository contract](../AGENTS.md).

- C++17, CUDA, OpenCL C API, and OpenCL C++ wrappers; CMake builds.
- Keep implementations in their existing backend directories and shared harnesses.
- Each executable supplies its correctness harness; availability varies by backend.
- Use `benchmark()` or `benchmarkWithReset()` for repeated timing, with warm-up,
  medians, and spread; use `benchmarkDevice()` for a kernel-only timing boundary
  alongside end-to-end. Identify device, precision, shape, and timing boundary.
- Existing [CUDA](../docs/optimization/cuda.md) and
  [OpenCL](../docs/optimization/opencl.md) guides govern future optimization work.
- Optional Python DSL backends (Triton, CuTe DSL) at `source/<kernel>/python/<dsl>/<kernel>.py`
  are standalone scripts, not CMake targets. They share a torch-based benchmarking helper,
  `source/utils/python/gpu_bench.py`, rather than `source/utils/benchmark.h`.
- AMD GPU support is planned through HIP for MI300X (`gfx942`), specified in
  [the HIP backend packet](2026-10-19-hip-backend/requirements.md). HIP is not
  implemented yet; CUDA remains the default build and CUDA/OpenCL remain the
  supported backends.
