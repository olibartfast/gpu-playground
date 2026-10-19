# Requirements: HIP/AMD backend support

Created: 2026-10-19. Status: planning only; no HIP implementation exists.
Branch: `feat/hip-backend` (proposed; not created).
Roadmap: [HIP/AMD backend](../roadmap.md).

Constraints: [mission](../mission.md), [technical boundaries](../tech-stack.md),
[repository contract](../../AGENTS.md).

## Goal

Add a first-class HIP backend for AMD Instinct MI300X (`gfx942`) that reuses the
existing backend-agnostic harnesses, proves correctness on one elementwise kernel
and one wave64-sensitive reduction kernel, and leaves CUDA and OpenCL behavior
unchanged.

## Findings

- F-1: the build selects exactly one backend family. Root
  [CMakeLists.txt](../../CMakeLists.txt) declares `LANGUAGES CXX CUDA` unless
  `USE_OPENCL` or `USE_OPENCL_CPP` is set, and `cmake_minimum_required` is 3.20.
  `source/utils/CMakeLists.txt` mirrors this with `cuda_helpers` or the OpenCL
  helper targets.
- F-2: per-kernel build files branch only on OpenCL/OpenCL C++/CUDA. For example
  [source/sigmoid/CMakeLists.txt](../../source/sigmoid/CMakeLists.txt) and the
  include switch in [source/sigmoid/main.cpp](../../source/sigmoid/main.cpp).
- F-3: the benchmark contract already separates end-to-end and device time.
  `benchmark()`/`benchmarkWithReset()` and `benchmarkDevice()` live in
  [source/utils/benchmark.h](../../source/utils/benchmark.h); the CUDA-only
  `categorical_cross_entropy` exposes `kernel_time_ms` and records CUDA events
  ([implementation](../../source/categorical_cross_entropy/cuda/categorical_cross_entropy.cpp)).
- F-4: existing CUDA sources encode 32-lane assumptions that must not be copied
  into HIP:
  - `WARP_SIZE = 32` in [source/spmv/cuda/spmv.h](../../source/spmv/cuda/spmv.h)
    and [source/categorical_cross_entropy/cuda/categorical_cross_entropy.cpp](../../source/categorical_cross_entropy/cuda/categorical_cross_entropy.cpp),
    with `0xffffffffu` shuffle masks.
  - `cooperative_groups` in
    [source/deep_learning_inference/cuda/dnn_inference.cpp](../../source/deep_learning_inference/cuda/dnn_inference.cpp).
- F-5: the local development environment on 2026-10-19 has no ROCm: `hipcc`,
  `rocminfo`, and `/opt/rocm` are absent. `clinfo` and `nvidia-smi` are present.
  AMD validation therefore requires the cloud path.
- F-6: the planning target is one MI300X on the AMD Developer Cloud
  (`gfx942`, 64-wide wavefronts). The image reports ROCm 6.4.0 in the planning
  input; the actual version, CMake, and `hipcc` must be recorded on first boot.
- F-7: native CMake HIP language support requires CMake 3.21 or newer, so the
  current 3.20 minimum must change if HIP is a first-class backend.
- F-8: the repository already has OpenCL C and OpenCL C++ backends. They remain
  supported, but they are not the chosen path for AMD kernel development.

## In Scope

- **R-1 — Build selection.** `USE_HIP` selects the HIP backend. It is mutually
  exclusive with `USE_OPENCL` and `USE_OPENCL_CPP`; the default remains CUDA.
  With `USE_HIP=ON`:
  - the project enables C++17 and the HIP language;
  - `CMAKE_HIP_ARCHITECTURES` is honored and defaults to `gfx942` only when unset;
  - `GPU_ENABLE_*` flags still control which kernels are built;
  - configuration fails with a clear message when the HIP toolchain is missing.
  Existing CUDA/OpenCL configure invocations must behave as before.
- **R-2 — Backend layout and API.** A HIP implementation lives at
  `source/<kernel>/hip/<kernel>.hip` with a backend header at
  `source/<kernel>/hip/<kernel>.h`.
  - The header exposes only the CPU reference and host-pointer API used by
    `main.cpp`; no HIP types leak into `main.cpp` or shared headers.
  - `main.cpp` selects it with `GPU_HIP_BACKEND`.
  - `source/utils/hip_helpers.h` provides `HIP_CHECK` with file, line, and error
    string, mirroring `CUDA_CHECK`; `source/utils/CMakeLists.txt` provides a
    `hip_helpers` interface target.
  - Host-pointer wrapper signatures match the CUDA/OpenCL backends.
- **R-3 — Harness and timing contract.** HIP backends preserve the existing
  correctness harness: CPU reference, stated tolerance, non-zero exit on
  mismatch. Allocation, transfers, and launches stay in the HIP implementation.
  Host timing uses `benchmark()`/`benchmarkWithReset()`. Kernels that expose
  `float* kernel_time_ms` use HIP events and `benchmarkDevice()`, so device time
  excludes allocation and transfers, matching the CUDA contract.
- **R-4 — Wave64 correctness.** HIP sources must not encode NVIDIA's 32-lane
  warp geometry:
  - lane and warp indices derive from `warpSize`, not a hardcoded 32;
  - shuffle and ballot masks are 64-bit where required, and ballot results are
    not truncated to 32 bits;
  - shared storage for per-warp partials is sized for the maximum supported
    wavefront, with the active warp count computed at runtime;
  - the wave64 policy is documented in `docs/optimization/hip.md`.
  CUDA sources are not rewritten; only the HIP copies carry this policy.
- **R-5 — Initial kernel coverage.** This increment ports exactly two kernels:
  - `sigmoid` — elementwise, scalar and vectorized variants; proves build, host
    wrapper, and correctness with no cross-thread communication.
  - `categorical_cross_entropy` — masked block reductions plus kernel-only
    timing; proves the wave64 policy and the `benchmarkDevice()` path.
  Other kernels remain CUDA/OpenCL only in this increment.
- **R-6 — Cloud validation evidence.** HIP builds target `gfx942` and run on one
  MI300X. Evidence records `amd-smi version`, `amd-smi static --asic`,
  `hipcc --version`, `cmake --version`, the image/container tag, and the date.
  Cross-vendor speedups are not claimed unless both sides are measured with the
  same shapes, precision, timing boundary, and stated hardware. A missing
  measurement is recorded as a gap, not a pass.
- **R-7 — Documentation.** When the backend lands:
  - `Readme.md` lists HIP as a supported backend, gives build/run commands, and
    marks HIP coverage in the kernel table;
  - `AGENTS.md` adds `hip/` to the backend split, the build matrix, the harness
    contract, and the optimization rules;
  - `docs/optimization/hip.md` covers `gfx` targets, wavefront width, HIPIFY
    limits, error handling, benchmarking, and ROCm profiling;
  - `docs/guides/adding-a-new-kernel.md` describes the HIP directory and CMake
    branch;
  - `specs/tech-stack.md` and `specs/roadmap.md` reflect the shipped state.
- **R-8 — No regression.** Without `USE_HIP`, CUDA and OpenCL configure/build/run
  paths remain unchanged. At least one CUDA harness and one OpenCL harness are
  rebuilt and run after the change. `GPU_ENABLE_*` flags continue to skip
  targets cleanly.

## Out of Scope

- Porting the remaining kernels. A later packet covers portable kernels in
  batches; `gemm` performance work is separate.
- `gemm` optimization, MFMA/rocWMMA/Composable Kernel integration, and any
  performance-parity claim with the CUDA backend.
- CUDA-only kernels that need cooperative groups (`deep_learning_inference`),
  and FP16/FP4-specific work.
- `gfx950`/MI355X, MI300A, `gfx90a`, RDNA/consumer GPUs, Windows, and the HIP
  SDK for Windows.
- Multi-GPU or 8x MI300X scaling, distributed execution, and orchestration.
- Replacing CUDA as the default backend or making HIP the canonical source for
  NVIDIA.
- Changing OpenCL backends, the CPU references, or `source/utils/benchmark.h`.
- CI with AMD GPU runners. A static ROCm-container compile job is a future idea.
- Committing `doctl` automation, cloud credentials, or cost-management scripts.
- Reworking shared CPU references into a common directory.

## Decisions

- **D-1 — HIP over OpenCL for AMD.** HIP is AMD's first-class programming model,
  keeps the CUDA mental model, and exposes wavefront intrinsics, MFMA, and the
  ROCm profilers. OpenCL remains a supported cross-vendor backend but is not
  extended for AMD. Rationale touches the backend boundaries in `tech-stack.md`.
- **D-2 — Explicit `hip/` directories, not a generated tree.** HIPIFY output is a
  draft; the committed `hip/` files are hand-reviewed and maintained. This fits
  the repository's backend split and allows wave64-specific fixes. Generating
  HIP from CUDA at configure time is rejected for now because it would hide
  those fixes.
- **D-3 — CUDA stays canonical and default.** AMD is an additional backend, not
  a replacement. This keeps the existing NVIDIA workflow and the OpenCL matrix
  intact while HIP matures.
- **D-4 — Target `gfx942` explicitly.** AMD has no PTX-like forward-compatible
  fallback, so the architecture list is explicit. Additional `gfx` targets are
  added only when hardware is available to validate them.
- **D-5 — CMake 3.21 minimum.** Native HIP language support starts there. The
  prerequisite documentation and constants that mention 3.20 must be updated.
- **D-6 — `.hip` translation units.** Kernel sources use the `.hip` extension;
  headers and `main.cpp` remain C++. This avoids setting `LANGUAGE HIP` as a
  source property and matches CMake's native HIP support.
- **D-7 — Reuse CPU references.** Each HIP implementation provides the same CPU
  reference and host-pointer wrapper as the existing backends. No shared
  `cpu/` refactor is part of this increment.
- **D-8 — Device timing uses HIP events.** The `kernel_time_ms` contract is
  implemented with `hipEventCreate`/`hipEventRecord`/`hipEventElapsedTime`, not
  a host clock around the launch.
- **D-9 — Elementwise first, wave64 reduction second.** `sigmoid` proves the
  build and harness path with minimal risk; `categorical_cross_entropy` then
  proves the wavefront policy and device timing.

## Constraints

- Follow the repository Hard Rules and Agent Rules in
  [AGENTS.md](../../AGENTS.md): backend directories, `*_CHECK` error handling,
  no device-side allocation, one optimization variable at a time, and no
  unvalidated performance claims.
- C++17 for host code. HIP code must compile with the ROCm toolchain shipped by
  the cloud image; do not require a newer ROCm locally.
- Correctness precedes timing. Record the device, ROCm version, `gfx` target,
  precision, shapes, and timing boundary with every measurement.
- Cloud cost discipline: do mechanical porting and review locally; use one
  MI300X for build/validate/profile; build only affected targets; copy evidence
  off the box; destroy the Droplet when idle. A powered-off Droplet still bills.
- No secrets in the repository. `doctl` tokens and SSH keys stay outside version
  control.

## Dependencies

- AMD Developer Cloud account, SSH access, and one `gpu-mi300x1-192gb` Droplet.
- ROCm with `hipcc`, CMake 3.21+, and the HIP runtime on the image.
- Local CUDA toolchain for regression builds (present on 2026-10-19) and an
  OpenCL runtime for the OpenCL regression build (`clinfo` is present).
- Existing backend-agnostic harnesses, CPU references, and benchmark helpers.
- [Benchmark device timing packet](../2026-10-01-benchmark-device-timing/requirements.md)
  for the `benchmarkDevice()` contract.

## Context

- The planned workflow follows the AMD Developer Cloud split: identify the GPU
  and `gfx` target, pin the stack, port and review locally, then build, validate,
  and profile on the cloud. The cloud image is not a free notebook tier; treat
  GPU hours as the scarce resource.
- The repository currently uses `cuda/`, `opencl/`, and `opencl_cpp/` per
  kernel, with a shared `main.cpp`. HIP follows the same shape.
- Existing OpenCL backends can serve as an independent comparison point on AMD
  if the image provides an ICD, but they are not a dependency of this packet.
- The 2026-10-19 planning discussion selected HIP over OpenCL and the MI300X
  cloud target. This packet records that direction; it does not claim any
  implementation or measurement.

## Assumptions & Open Questions

- [A-1] The cloud image provides `hipcc`, CMake 3.21+, and a ROCm version that
  supports `gfx942`. Basis: planning input; confirm on first boot.
- [A-2] `gfx942` is the only architecture needed for this increment. Basis: the
  target droplet is MI300X; no other AMD hardware is available.
- [A-3] The existing CPU references and tolerances are adequate for HIP without
  numerical changes. Basis: HIP and CUDA use the same IEEE float operations for
  these kernels; confirm with the actual runs.
- [A-4] The existing `GPU_ENABLE_*` flags are sufficient to keep unported
  kernels out of the HIP build. Basis: root CMake already gates each subdirectory.
- [Q-1] After the foundation, should coverage prioritize elementwise/permutation
  kernels or reductions? Recommendation: finish the portable kernels in
  Readme order, leaving `gemm` performance for a dedicated packet. Owner: next
  planning session.
- [Q-2] Should HIP eventually become the canonical source for both vendors via
  HIP's NVIDIA platform? Not in this increment; revisit after coverage stabilizes.
- [Q-3] Should CI gain a ROCm-container compile job without a GPU? Deferred;
  it would catch configuration drift but not correctness.
- [Q-4] Should `gfx950`/MI355X support be added when the Developer Cloud offers
  it? Deferred until the hardware is available; the architecture list is
  explicit by design.
- [Q-5] Is consumer RDNA support expected? Current assumption is no; the
  wavefront width differs and would need its own validation.

## Definition of Done (requirements level)

- [ ] Every [R-n] is implemented or explicitly deferred with a tracked location
- [ ] No In Scope behavior silently dropped; no Out of Scope work smuggled in
- [ ] CUDA and OpenCL regression builds pass
- [ ] HIP evidence from `gfx942` is recorded with device, ROCm, compiler, and date
- [ ] Docs and roadmap agree with what actually shipped
