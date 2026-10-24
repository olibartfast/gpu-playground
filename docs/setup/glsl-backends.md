# GLSL Compute Backends

> Audience: humans, agents · Applies to: OpenGL, Vulkan · Last reviewed: 2026-10

The initial GLSL coverage is sigmoid only, with scalar and four-elements-per-invocation
variants. Both APIs compile the same `source/sigmoid/glsl/sigmoid.comp` source,
use 128 threads per workgroup, and validate against the same FP32 CPU reference.
Other kernels are explicitly skipped at configure time.

## Build and run

On Ubuntu, install development dependencies (CUDA is not required):

```bash
sudo apt install cmake ninja-build g++ libegl-dev libgl-dev libvulkan-dev glslang-tools
cmake --preset opengl
cmake --build --preset opengl -j4
ctest --preset opengl
cmake --preset vulkan
cmake --build --preset vulkan -j4
ctest --preset vulkan
```

A working graphics driver is also required. OpenGL requires a 4.3 core context
and EGL surfaceless support; it first tries the Mesa surfaceless platform, then
the default EGL display. No window is created. Vulkan requires a compute queue
with timestamps, 128-thread workgroups, and host-visible storage-buffer memory.
Vulkan prefers discrete GPUs over integrated GPUs and software devices. Check
both printed device names before comparing; OpenGL follows EGL's device selection.
If necessary, use driver environment settings to select matching devices.

The options `USE_OPENGL` and `USE_VULKAN` are mutually exclusive with each other
and with HIP/OpenCL options. They honor the existing `GPU_ENABLE_*` flags.
Python CUDA tests are not registered for either GLSL backend.

Run sequentially to avoid contention:

```bash
./build/opengl/source/sigmoid/sigmoid | tee /tmp/sigmoid-opengl.log
./build/vulkan/source/sigmoid/sigmoid | tee /tmp/sigmoid-vulkan.log
```

Vulkan shaders are compiled to SPIR-V at build time using glslangValidator.
Override its location with `-DGLSLANG_VALIDATOR=/path/to/glslangValidator`.
OpenGL compiles GLSL at initialization and surfaces compilation/link logs.
Both binaries use absolute build/source paths to their shader assets and can
run from another working directory. Keep the source shader for OpenGL and the
build's `.spv` files for Vulkan; these are build-tree executables, not installed packages.

## Measurement contract

Each executable checks sizes 0, 1, 257, 1024 and 1000003, rejects nonfinite
results, and uses absolute tolerance 1e-5. Zero size is a no-op, not a dispatch
benchmark. Each variant is checked before benchmarking and after the final
iteration. CPU and GPU paths use 3 warm-up and 10 measured calls, reporting median
and spread through `benchmark.h`. Validation and output poisoning are untimed.

- `GPU end-to-end:` includes buffer allocation/destruction, upload, command setup,
  dispatch, completion wait, timestamp retrieval and readback. Device/context and
  shader/pipeline creation are cached and excluded from the measured iterations.
- `GPU kernel:` is a device timestamp interval around a dispatch. OpenGL uses
  `GL_TIMESTAMP` queries; Vulkan uses top-of-pipe/bottom-of-pipe timestamps with
  the queue's valid-bit mask and device timestamp period. These intervals include
  API/driver scheduling effects and are not guaranteed identical instruction-level boundaries.
- OpenGL uses driver-managed storage buffers and upload/readback calls. Vulkan
  directly maps host-visible allocations, preferring coherent memory and explicitly
  flushing/invalidating noncoherent memory. This is a baseline host-memory policy;
  dedicated device-local allocations with staging are a future experiment.

Compare matching sizes and variants on the same GPU. Treat end-to-end results
as measurements of these implementations, including their different allocation
and memory policies, rather than a universal API ranking. The four-element
variant uses scalar buffer accesses; it does not guarantee hardware vector loads.
No optimization or speedup is assumed.

For Vulkan correctness diagnostics, optionally install `vulkan-validationlayers`:

```bash
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./build/vulkan/source/sigmoid/sigmoid
```

Do performance runs without validation layers. Synchronization follows the
[Khronos host-readback example](https://docs.vulkan.org/guide/latest/synchronization_examples.html).
