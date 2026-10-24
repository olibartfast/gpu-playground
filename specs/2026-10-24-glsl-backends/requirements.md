# GLSL compute backends

R1: Add mutually exclusive USE_OPENGL and USE_VULKAN options and presets without requiring CUDA. Keep existing backends intact; unported kernels skip explicitly.
R2: Start both backends with sigmoid scalar and four-elements-per-invocation variants from one GLSL source. No other kernel ports in this increment.
R3: Validate finite output against CPU at absolute tolerance 1e-5 for sizes 0, 1, 257, 1024, 1000003, covering tails and large inputs.
R4: Report device identity and repeated CPU, end-to-end, and device timestamp timings using benchmark.h. Context/device and pipeline initialization are cached and excluded after warm-up; allocation, upload, dispatch, synchronization and readback are included in end-to-end. No performance superiority assumed.
R5: Document dependencies, headless execution, backend coverage and reproducible comparisons. Preserve backend-agnostic harness boundaries.

Existing mission and measurement conventions apply. Both APIs were explicitly requested for benchmarking. Full kernel coverage and graphics rendering are deferred.
