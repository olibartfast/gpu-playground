# Plan: HIP/AMD Backend Foundation

Requirements: [requirements.md](requirements.md).
Validation is defined in [validation.md](validation.md) before implementation.
Status: planning only. No task in this packet has been executed.

## Phase 0 — Freeze Scope and Acceptance (This Change)

- **Freeze the packet (T-0; R-1–R-8).** Create this packet, update the
  [roadmap](../roadmap.md) and [tech stack](../tech-stack.md), and freeze the
  commands in `validation.md`.
  - Deliverable: a reviewable specification with explicit in-scope and
    out-of-scope boundaries.
  - Checks: `git diff --check`; every R-n requirement in `requirements.md` has a
    V-n row; every path link resolves.

Gate: planning checks are recorded in `validation.md`. No implementation begins
until the packet is reviewed.

## Phase 1 — Build-System Scaffold

- **Add `USE_HIP` to the root CMake (T-1; R-1, R-8).**
  - Make `USE_HIP`, `USE_OPENCL`, and `USE_OPENCL_CPP` mutually exclusive, with
    a clear configure error when more than one is set.
  - When `USE_HIP=ON`, declare `project(... LANGUAGES CXX HIP)`, set C++17, and
    default `CMAKE_HIP_ARCHITECTURES` to `gfx942` only if it is unset.
  - Bump `cmake_minimum_required` to 3.21 and update the prerequisites text.
  - Keep the CUDA and OpenCL branches behaviorally equivalent.
  - Deliverable: a HIP configure path exists, and the CUDA and OpenCL configure
    paths still work.
- **Add the HIP helper (T-2; R-2).** Add `source/utils/hip_helpers.h` and a
  `hip_helpers` interface target in `source/utils/CMakeLists.txt`.
  - `HIP_CHECK` prints the file, line, and `hipGetErrorString` result, then
    aborts.
  - Link and include the HIP runtime through the target, not through `main.cpp`.
  - Deliverable: a header usable by a HIP translation unit, with no host-side
    plumbing in the harness.
- **Add a HIP preset (T-3; R-1).** Add a `hip` configure and build preset in
  `CMakePresets.json` (`binaryDir: build/hip`, `USE_HIP=ON`, `gfx942`, Release)
  and leave the existing presets untouched.
  - Deliverable: `cmake --preset hip` is the documented cloud entry point.
- **Gate kernel subdirectories for HIP (T-4; R-1, R-8).**
  - Ported kernels get a `USE_HIP` branch. Unported kernels must be skipped by
    the root CMake, or return early in their own `CMakeLists.txt`, when
    `USE_HIP` is set. They must not fall through to the CUDA branch.
  - They keep their CUDA and OpenCL builds unchanged when `USE_HIP` is off.
  - Deliverable: a default HIP configure succeeds with only the kernels claimed
    by R-5.

Gate: V-2 on the local CUDA/OpenCL matrix, and V-3 on `gfx942`. A HIP configure
failure that leaves CUDA and OpenCL untouched is a local phase failure, not a
reason to edit existing kernel math.

## Phase 2 — `sigmoid` Proof

- **Create the HIP sigmoid backend (T-5; R-2, R-3, R-5).** Create
  `source/sigmoid/hip/sigmoid.h` and `source/sigmoid/hip/sigmoid.hip`.
  - Start from `hipify-perl` output for the CUDA file, then hand-review every API
    call, include, and launch. Keep the scalar and vectorized variants.
  - Provide the CPU reference and the same host-pointer wrappers as the other
    backends. Allocate, copy, launch, and free inside the wrapper.
  - Use `HIP_CHECK` on every runtime call and check the launch.
  - Deliverable: a HIP backend that compiles and is callable through the
    existing `main.cpp`.
- **Wire the harness (T-6; R-2).** Update `source/sigmoid/CMakeLists.txt` with
  the `USE_HIP` branch, and `source/sigmoid/main.cpp` with `GPU_HIP_BACKEND`
  include selection.
  - Deliverable: `--target sigmoid` builds with `USE_HIP=ON` and still builds
    with CUDA and OpenCL.
- **Validate on the MI300X (T-7; R-6).** Configure, build, run the harness, and
  record the environment and timings.
  - Deliverable: a correctness pass for both variants, plus a benchmark line
    with warm-up and median. Evidence is appended to `validation.md`.

Gate: V-3, V-4, and V-5 for `sigmoid`. If correctness fails, stop and fix the
port before adding the reduction kernel.

## Phase 3 — Wave64 Reduction Proof

- **Write the wave64 policy first (T-8; R-4).** Write the wave64 section of
  `docs/optimization/hip.md` before porting: lane and warp derivation from
  `warpSize`, 64-bit masks, ballot width, shared-storage sizing, and the
  `spmv`/`categorical_cross_entropy` audit list.
  - Deliverable: a policy against which T-9 can be reviewed.
- **Create the HIP reduction kernel (T-9; R-3, R-4, R-5).** Create
  `source/categorical_cross_entropy/hip/` from the CUDA implementation.
  - Replace the 32-lane constants: derive the active warp count from `warpSize`,
    size shared storage for the maximum supported wavefront, use 64-bit shuffle
    masks, and do not truncate ballots.
  - Implement `kernel_time_ms` with HIP events, keeping allocation and
    transfers outside the timed region.
  - Preserve the `solve` entry point and all functional test cases.
  - Deliverable: a HIP translation unit whose reduction is correct at `gfx942`.
- **Wire the build and harness (T-10; R-5, R-8).** Update the kernel's CMake and
  the harness include branch, and remove the "CUDA-only" exclusion for HIP in the
  root build.
  - Deliverable: `--target categorical_cross_entropy` builds under HIP, and the
    CUDA build still resolves the CUDA sources.
- **Validate and profile (T-11; R-6).** Run the functional cases and the
  `--performance` shape on the MI300X. Run
  `rocprofv3 --kernel-trace --stats` once and confirm that the expected kernels
  appear.
  - Deliverable: PASS lines for every case, device and end-to-end timings, and a
    trace summary. Evidence is appended to `validation.md`.

Gate: V-4, V-5, V-6, and V-8 review. The wave64 policy is accepted only after
the reduction runs correctly on a 64-wide wavefront.

## Phase 4 — Documentation, Regression, and Integration

- **Update the documentation (T-12; R-7).** Update `Readme.md`, `AGENTS.md`,
  `docs/README.md`, `docs/guides/adding-a-new-kernel.md`,
  `docs/optimization/hip.md`, `specs/tech-stack.md`, and `specs/roadmap.md`.
  - Readme: HIP in the backend list, build and run commands, and HIP coverage
    markers.
  - AGENTS: `hip/` in the backend split, build matrix, harness contract, and
    optimization rules. Add an `hip-optimizer` role only if the existing roles
    cannot express the work.
  - New-kernel guide: the HIP directory, `.hip` extension, `HIP_CHECK`, and the
    wave64 checklist.
  - Deliverable: a reader can build, run, and extend the HIP backend without
    reading this packet.
- **Check for regressions (T-13; R-8).** Rebuild and run one CUDA harness and
  one OpenCL harness. Run `git diff --check`, and confirm that no credentials or
  cloud artifacts are tracked.
  - Deliverable: regression evidence in `validation.md`.
- **Retire the cloud resources (T-14; R-6).** rsync results off the Droplet,
  snapshot only if it saves setup time, and destroy the Droplet. Record the
  lifecycle action, not the token.
  - Deliverable: evidence retained locally, with no running billable instance.

Gate: V-1, V-2, V-7, V-8, and the integration Definition of Done.

## Follow-On Work (Separate Packets)

These are roadmap items, not commitments of this packet:

1. **Portable kernel coverage.** Elementwise, permutation, reduction, sparse,
   and convolution kernels in small batches, each with its own validation.
   `deep_learning_inference` waits for cooperative-groups support.
2. **GEMM performance.** rocWMMA/MFMA tiling, LDS layout, and rocprof-compute
   evidence. No performance claim until measured against a HIP baseline.
3. **Cloud and CI.** An optional ROCm-container compile job without a GPU; no
   assumption of GPU runners.
4. **Additional targets.** `gfx950`/MI355X or RDNA only when hardware is
   available to validate them.

## Integration Checkpoint

One pull request from `feat/hip-backend` contains this packet, the code, and the
documentation. The branch is rebased onto `master` before review. On merge,
update the roadmap status with evidence dates. If a phase is cut, record the
deferral in `requirements.md` and the roadmap rather than leaving a silent gap.
