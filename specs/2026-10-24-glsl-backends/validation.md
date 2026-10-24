# Validation

Defined before implementation.

- R1: configure/build both presets; reject mixed backend options; confirm only sigmoid registers in each CTest suite.
- R2/R3: run CTest for both APIs over all specified sizes and variants; validate SPIR-V.
- R4: record device, driver, median/spread, timing boundaries; compare only the same physical device.
- R5: inspect docs and run git diff --check; rebuild/run existing OpenCL sigmoid.

Baseline: existing OpenCL sigmoid scalar and vectorized output matched CPU on 2026-10-24.

## Executed 2026-10-24

- R1: `cmake --preset opengl`, `cmake --build --preset opengl -j4`,
  `cmake --preset vulkan -DGLSLANG_VALIDATOR=/tmp/gpu-glsl-tools/usr/bin/glslangValidator`,
  `cmake --build --preset vulkan -j4`: passed. Each backend registers exactly one CTest, sigmoid.
- R1 negative check: `cmake -S . -B /tmp/gpu-glsl-conflict -DUSE_OPENGL=ON -DUSE_VULKAN=ON`
  rejected as expected, before compiler detection.
- R2/R3: `ctest --preset opengl` and `ctest --preset vulkan`: passed all five sizes and both variants.
  `spirv-val --target-env vulkan1.1 build/vulkan/source/sigmoid/sigmoid_{1,4}.spv`
  (invoked separately per file): both passed.
- Vulkan validation: ran with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`,
  locally unpacked validation layers 1.3.275.0; exit 0 and no validation diagnostics.
- R4: both selected NVIDIA GeForce RTX 3060 Laptop GPU, NVIDIA driver 610.43.02;
  OpenGL 4.3 context, Vulkan device API 1.4. GCC 13.3.0, RelWithDebInfo;
  Vulkan headers/loader development version 1.3.275, glslang-tools 15.1.0,
  SPIR-V Tools 2025.1. No validation layers in performance runs.
- R5: `python3 .github/scripts/check-docs.py`, `git diff --check`: passed.
  Existing CUDA sigmoid rebuilt and both variants matched CPU. Existing OpenCL
  sigmoid rebuilt and both variants matched CPU before implementation.

Sequential standalone baseline runs, 3 warm-up + 10 measured iterations,
N=1000003, FP32. Raw results include spread and all sizes:
[OpenGL](evidence/opengl.txt), [Vulkan](evidence/vulkan.txt).

| API | Variant | End-to-end median ms | Dispatch timestamp median ms |
|---|---|---:|---:|
| OpenGL | scalar | 13.1305 | 0.0307 |
| OpenGL | four elements | 13.0082 | 0.0369 |
| Vulkan | scalar | 43.1062 | 0.7159 |
| Vulkan | four elements | 105.1989 | 60.5241 |

These are baseline implementation measurements, not a universal API comparison.
Vulkan's four-element variant has a large measured regression; its cause has not
been profiled. Direct host-visible Vulkan storage and OpenGL driver-managed
storage are different memory policies. No device-local staging or optimization
claim is included. GPU clocks/power were not locked; no cross-vendor validation,
HIP regression, full-repository test suite, or automated graphics CI was run.

Local tools needed no administrator installation: `apt-get download glslang-tools
spirv-tools vulkan-validationlayers` in `/tmp/gpu-glsl-tools`, followed by
`dpkg-deb -x` per package into that directory. The validation run set
`VK_LAYER_PATH=/tmp/gpu-glsl-tools/usr/share/vulkan/explicit_layer.d` and
`LD_LIBRARY_PATH=/tmp/gpu-glsl-tools/usr/lib/x86_64-linux-gnu`.
