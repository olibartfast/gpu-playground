# Technical boundaries

Evidence: [CMake](../CMakeLists.txt) and [repository contract](../AGENTS.md).

- C++17, CUDA, OpenCL C API, and OpenCL C++ wrappers; CMake builds.
- Keep implementations in their existing backend directories and shared harnesses.
- Each executable supplies its correctness harness; availability varies by backend.
- Use `benchmark()` or `benchmarkWithReset()` for repeated timing, with warm-up,
  medians, and spread. Identify device, precision, shape, and timing boundary.
- Existing [CUDA](../docs/cuda-agent-guide.md) and
  [OpenCL](../docs/opencl-agent-guide.md) guides govern future optimization work.
