# Plan: HIP/AMD backend foundation

Requirements: [requirements.md](requirements.md).
Validation is defined in [validation.md](validation.md) before implementation.
Status: planning only. No task in this packet has been executed.

## Phase 0 — Freeze scope and acceptance (this change)

- **T-0 (R-1–R-8):** create this packet, update the [roadmap](../roadmap.md) and
  [tech stack](../tech-stack.md), and freeze the commands in `validation.md`.
  Deliverable: reviewable specification with explicit in/out scope.
  - Checks: `git diff --check`; every `R-n` in requirements has a `V-n` row;
    every path link resolves.

Gate: planning checks recorded in `validation.md`. No implementation begins
until the packet is reviewed.

## Phase 1 — Build-system scaffold (walk)

- **T-1 (R-1, R-8):** add `USE_HIP` to the root `CMakeLists.txt`.
  - Make `USE_HIP`, `USE_OPENCL`, and `USE_OPENCL_CPP` mutually exclusive with a
    clear configure error when more than one is set.
  - When `USE_HIP=ON`, declare `project(... LANGUAGES CXX HIP)`, set C++17, and
    default `CMAKE_HIP_ARCHITECTURES` to `gfx942` only if it is unset.
  - Bump `cmake_minimum_required` to 3.21 and update the prerequisites text.
  - Keep the CUDA and OpenCL branches byte-for-byte equivalent in behavior.
  - Deliverable: HIP configure path exists; CUDA/OpenCL configure still works.
- **T-2 (R-2):** add `source/utils/hip_helpers.h` and a `hip_helpers` interface
  target in `source/utils/CMakeLists.txt`.
  - `HIP_CHECK` prints file, line, and `hipGetErrorString` and aborts.
  - Link/include the HIP runtime through the target, not through `main.cpp`.
  - Deliverable: a header usable by a HIP translation unit with no host-side
    plumbing in the harness.
- **T-3 (R-1):** add a `hip` configure/build preset in `CMakePresets.json`
  (`binaryDir: build/hip`, `USE_HIP=ON`, `gfx942`, Release) and keep the existing
  presets untouched.
  - Deliverable: `cmake --preset hip` is the documented cloud entrypoint.
- **T-4 (R-1, R-8):** gate per-kernel subdirectories for HIP.
  - Ported kernels get a `USE_HIP` branch. Unported kernels must be skipped by
    the root CMake (or return early in their own `CMakeLists.txt`) when
    `USE_HIP` is set; they must not fall through to the CUDA branch.
  - They keep their CUDA/OpenCL builds unchanged when `USE_HIP` is off.
  - Deliverable: a default HIP configure succeeds with only the kernels claimed
    by R-5.

Gate: V-2 on the local CUDA/OpenCL matrix and V-3 on `gfx942`. A failure in the
HIP configure that leaves CUDA/OpenCL untouched is a local phase failure, not a
reason to edit existing kernel math.

## Phase 2 — `sigmoid` proof (run)

- **T-5 (R-2, R-3, R-5):** create `source/sigmoid/hip/sigmoid.h` and
  `source/sigmoid/hip/sigmoid.hip`.
  - Start from `hipify-perl` output for the CUDA file; hand-review every API call,
    include, and launch. Keep the scalar and vectorized variants.
  - Provide the CPU reference and the same host-pointer wrappers as the other
    backends; allocate, copy, launch, and free inside the wrapper.
  - Use `HIP_CHECK` on every runtime call and check the launch.
  - Deliverable: a HIP backend that compiles and is callable through the
    existing `main.cpp`.
- **T-6 (R-2):** update `source/sigmoid/CMakeLists.txt` with the `USE_HIP` branch
  and `source/sigmoid/main.cpp` with `GPU_HIP_BACKEND` include selection.
  - Deliverable: `--target sigmoid` builds with `USE_HIP=ON` and still builds
    with CUDA and OpenCL.
- **T-7 (R-6):** on the MI300X, configure, build, run the harness, and record the
  environment and timings.
  - Deliverable: correctness pass for both variants plus a benchmark line with
    warm-up and median; evidence appended to `validation.md`.

Gate: V-3, V-4, V-5 for `sigmoid`. If correctness fails, stop and fix the port
before adding the reduction kernel.

## Phase 3 — Wave64 reduction proof (run)

- **T-8 (R-4):** write the wave64 section of `docs/optimization/hip.md` before
  porting: lane/warp derivation from `warpSize`, 64-bit masks, ballot width,
  shared-storage sizing, and the `spmv`/`categorical_cross_entropy` audit list.
  - Deliverable: a policy that T-9 can be reviewed against.
- **T-9 (R-3, R-4, R-5):** create `source/categorical_cross_entropy/hip/` from
  the CUDA implementation.
  - Replace the 32-lane constants: derive the active warp count from
    `warpSize`, size shared storage for the maximum supported wavefront, use
    64-bit shuffle masks, and do not truncate ballots.
  - Implement `kernel_time_ms` with HIP events; keep allocation and transfers
    outside the timed region.
  - Preserve the `solve` entrypoint and all functional test cases.
  - Deliverable: a HIP translation unit whose reduction is correct at `gfx942`.
- **T-10 (R-5, R-8):** update the kernel CMake and harness include branch, and
  remove the "CUDA-only" exclusion for HIP in the root build.
  - Deliverable: `--target categorical_cross_entropy` builds under HIP and the
    CUDA build still resolves the CUDA sources.
- **T-11 (R-6):** run the functional cases and `--performance` shape on the
  MI300X. Run `rocprofv3 --kernel-trace --stats` once and record that the
  expected kernels appear.
  - Deliverable: PASS lines for every case, device/end-to-end timings, and a
    trace summary; evidence appended to `validation.md`.

Gate: V-4, V-5, V-6, and V-8 review. The wave64 policy is accepted only after
the reduction runs correctly on a 64-wide wavefront.

## Phase 4 — Documentation, regression, integration

- **T-12 (R-7):** update `Readme.md`, `AGENTS.md`, `docs/README.md`,
  `docs/guides/adding-a-new-kernel.md`, `docs/optimization/hip.md`,
  `specs/tech-stack.md`, and `specs/roadmap.md`.
  - Readme: HIP in the backend list, build/run commands, HIP coverage markers.
  - AGENTS: `hip/` in the backend split, build matrix, harness contract, and
    optimization rules; add an `hip-optimizer` role only if the existing roles
    cannot express the work.
  - New-kernel guide: HIP directory, `.hip` extension, `HIP_CHECK`, and wave64
    checklist.
  - Deliverable: a reader can build, run, and extend the HIP backend without
    reading this packet.
- **T-13 (R-8):** rebuild and run one CUDA harness and one OpenCL harness; run
  `git diff --check`; confirm no credentials or cloud artifacts are tracked.
  - Deliverable: regression evidence in `validation.md`.
- **T-14 (R-6):** rsync results off the Droplet, snapshot only if it saves setup
  time, and destroy the Droplet. Record the lifecycle action, not the token.
  - Deliverable: evidence retained locally; no running billable instance.

Gate: V-1, V-2, V-7, V-8 and the integration Definition of Done.

## Follow-on work (separate packets)

These are roadmap items, not commitments of this packet:

1. **Portable kernel coverage:** elementwise, permutation, reduction, sparse,
   and convolution kernels in small batches, each with its own validation.
   `deep_learning_inference` waits for cooperative-groups support.
2. **GEMM performance:** rocWMMA/MFMA tiling, LDS layout, and rocprof-compute
   evidence; no performance claim until measured against a HIP baseline.
3. **Cloud and CI:** optional ROCm-container compile job without a GPU; no
   GPU-runner assumption.
4. **Additional targets:** `gfx950`/MI355X or RDNA only when hardware is
   available to validate them.

## Integration checkpoint

One PR from `feat/hip-backend` containing this packet, the code, and the docs.
The branch is rebased onto `master` before review. On merge, update the roadmap
status with evidence dates. If a phase is cut, record the deferral in
`requirements.md` and the roadmap rather than leaving a silent gap.
