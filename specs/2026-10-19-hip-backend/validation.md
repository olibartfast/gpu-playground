# Validation: HIP/AMD Backend Foundation

Status: specified 2026-10-19, before implementation. No implementation check has
been executed. The planning evidence at the end is explicitly not evidence that
the HIP backend works.

Requirements: [requirements.md](requirements.md). Plan: [plan.md](plan.md).

## Frozen Acceptance Commands

These commands are frozen for the implementation attempt. Changing one requires
a recorded reason in this file.

### Local Regression (CUDA and OpenCL)

Run from the repository root on the NVIDIA development machine:

```bash
cmake --preset native
cmake --build --preset native -j"$(nproc)"
./build/native/source/sigmoid/sigmoid

cmake -B build/opencl -DUSE_OPENCL=ON
cmake --build build/opencl -j"$(nproc)"
./build/opencl/source/sigmoid/sigmoid
```

Expected: both builds exit 0, and both binaries print their match message and
exit 0. Existing `build/` directories may be reused, but at least one of the two
backends must use a clean configure.

### Cloud HIP Validation

Run from the repository root on one MI300X (`gfx942`), after copying the branch
to the Droplet:

```bash
export PATH=/opt/rocm/bin:$PATH

cmake -S . -B build/hip -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build/hip --target sigmoid -j8
./build/hip/source/sigmoid/sigmoid

cmake --build build/hip --target categorical_cross_entropy -j8
./build/hip/source/categorical_cross_entropy/categorical_cross_entropy
./build/hip/source/categorical_cross_entropy/categorical_cross_entropy --performance

rocprofv3 --kernel-trace --stats -- \
  ./build/hip/source/categorical_cross_entropy/categorical_cross_entropy
```

Expected: configure and both builds exit 0; both harnesses exit 0; the
`categorical_cross_entropy` output contains PASS for every functional case and
for the performance case; and the profiler run lists the HIP kernels. Record the
exact `amd-smi` and `hipcc` output with the evidence.

## Baseline Expectations

- Before this packet: clean `master` at `959ebde`, with no `USE_HIP` option, no
  `hip/` directories, and no HIP helper.
- The local CUDA toolchain and OpenCL runtime are present; ROCm is absent.
  Therefore the CUDA/OpenCL regression is locally runnable, and the HIP path is
  not.
- V-2 must run on the unmodified tree at the start of Phase 1 and again after
  the changes. The first run is the regression baseline; the second proves no
  regression. Record both results, not just the final one.
- No timing baseline is claimed for HIP, because no HIP binary exists yet.

## Automated Checks

| ID | Check | Requirements | Observable evidence |
|---|---|---|---|
| V-1 | Static wiring | R-1, R-2, R-8 | `USE_HIP` and the mutual-exclusion error exist in the root CMake; `GPU_HIP_BACKEND` selects the HIP header; `source/utils/hip_helpers.h` defines `HIP_CHECK`; the `hip` preset exists; no unported kernel has a HIP target. `git diff --check` passes |
| V-2 | Local CUDA and OpenCL regression | R-1, R-8 | The local CUDA (`native` preset, so the build matches the local GPU) and OpenCL configure/build/run commands above succeed. The HIP branch is not required locally; its absence must not change either result |
| V-3 | HIP `sigmoid` build and run | R-1, R-5 | On `gfx942`, the HIP configure and `sigmoid` build succeed, and the binary prints `[kernel1] Results match!` and `[kernel2] Results match!` and exits 0 |
| V-4 | Wave64 reduction correctness | R-4, R-5 | On `gfx942`, `categorical_cross_entropy` prints PASS for every functional case and the `--performance` case. A grep of `source/*/hip` finds no hardcoded 32-lane geometry (`WARP_SIZE = 32`, `0xffffffffu` masks, 32-bit ballot truncation); manual review confirms `warpSize`-derived indices and 64-bit masks |
| V-5 | Timing evidence | R-3, R-6 | `sigmoid` output uses the `benchmark()` path with the default 3 warm-up and 10 timed iterations. `categorical_cross_entropy` prints `GPU kernel:` and `GPU end-to-end:` lines, and `kernel_time_ms` is produced by HIP events. Evidence records `gfx942`, ROCm, `hipcc`, CMake, the image tag, and the date |
| V-6 | `rocprofv3` trace | R-6 | `rocprofv3 --kernel-trace --stats` completes on the `categorical_cross_entropy` binary and lists the expected kernel names. No roofline, throughput, or cross-vendor speedup claim is made from this check |
| V-7 | Documentation review | R-7 | The Readme, AGENTS, `docs/optimization/hip.md`, and `docs/guides/adding-a-new-kernel.md` describe the HIP build, backend layout, wave64 rules, and the `HIP_CHECK` contract. Internal links resolve. `specs/tech-stack.md` and `specs/roadmap.md` match the shipped state |
| V-8 | Scope and security review | all | Read-only review: changed paths are limited to the packet, build files, `main.cpp` include branches, HIP sources, the helper, docs, and the roadmap. No CUDA or OpenCL kernel math changed; no credentials, Droplet IDs, or cloud artifacts are tracked. The scope diff confirms that the out-of-scope list stayed out: no additional kernel ports, no `gemm`/MFMA work, no `gfx950`/RDNA, no multi-GPU, no OpenCL changes, and no `benchmark.h` changes. The review verdict is recorded |

## Manual Checks

- [ ] **M-1 — First-boot stack.** Run `amd-smi version`, `amd-smi static --asic`,
  `rocminfo | grep -E "Marketing Name|Name:.*gfx"`, `hipcc --version`, and
  `cmake --version`. Confirm `gfx942` and CMake 3.21 or newer before building.
  Record the outputs; do not assume the image versions.
- [ ] **M-2 — Harness detects a HIP error.** In a throwaway copy of the tree,
  perturb the HIP `sigmoid` kernel by a deterministic factor (for example,
  multiply the output by `1.001`). The existing `main.cpp` must print a mismatch
  and exit non-zero. Restore the copy before continuing. This proves the harness
  is not passing vacuously.
- [ ] **M-3 — Repeatability.** Run each harness twice. Correctness must be
  identical; report the median and spread rather than a single timing sample.
- [ ] **M-4 — Device mismatch is legible.** If time permits, build once with
  `-DCMAKE_HIP_ARCHITECTURES=gfx950` and run on `gfx942`. Record the
  `hipErrorNoBinaryForGpu` or invalid-device-function message. This is a
  diagnostic, not a required acceptance check.
- [ ] **M-5 — Cloud lifecycle.** Copy evidence off the Droplet, snapshot only if
  it saves setup time, and destroy the Droplet. Confirm that no running billable
  instance remains. Do not record the `doctl` token.

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|---|---|---|---|---|
| P-1 | Read the `apply-spec-driven-development` skill and its artifact templates, interviewing, thin-phase, validation, and brownfield references | pass | 2026-10-19 | Packet follows the repository's existing spec style |
| P-2 | Inventory `source/*` backend directories and per-kernel CMake | pass | 2026-10-19 | 19 C++ kernels plus `vector_addition` (Python-only). 15 build with OpenCL; 4 are CUDA-only in the build: `categorical_cross_entropy`, `deep_learning_inference` (OpenCL source exists but CMake skips it), `fp16_dot_product`, `gaussian_blur` |
| P-3 | Inspect root CMake, presets, utils, sigmoid, categorical, spmv, softmax, prefix_sum, and new-kernel guide | pass | 2026-10-19 | Findings F-1–F-4, F-7 recorded in requirements |
| P-4 | Check local toolchain: `command -v hipcc`, `command -v rocminfo`, `ls /opt/rocm` | pass | 2026-10-19 | All absent; no local ROCm. `clinfo` and `nvidia-smi` present, so only CUDA/OpenCL regression is locally executable |
| P-5 | `git status --short`, `git branch --show-current`, `git log -1` | pass | 2026-10-19 | Clean tree on `master` at `959ebde` before this packet |
| P-6 | `git diff --check`; relative-link check across `specs/**/*.md` | pass | 2026-10-19 | `git diff --check` exit 0; 21 spec files checked, 0 broken relative links. Working tree before the packet: clean `master` at `959ebde`; after: only this packet and the two constitution edits |
| V-1 | Static wiring | not run | — | No implementation exists; planned for Phase 1 |
| V-2 | Local CUDA/OpenCL regression | not run | — | Planned for Phase 1 and Phase 4; no local code changes yet |
| V-3 | HIP `sigmoid` build/run | not run | — | Requires `gfx942` cloud access; planned for Phase 2 |
| V-4 | Wave64 reduction correctness | not run | — | Requires `gfx942` cloud access; planned for Phase 3 |
| V-5 | Timing evidence | not run | — | Requires HIP implementation and cloud access |
| V-6 | `rocprofv3` trace | not run | — | Requires HIP implementation and cloud access |
| V-7 | Documentation review | not run | — | Docs are part of Phase 4 |
| V-8 | Scope/security review | not run | — | Part of the future implementation PR |
| M-1 | First-boot stack inspection | not run | — | Planned for Phase 2 |
| M-2 | Fault-injected HIP mismatch | not run | — | Planned for Phase 2 |
| M-3 | Repeatability | not run | — | Planned for Phase 3 |
| M-4 | Wrong-architecture diagnostic | not run | — | Optional |
| M-5 | Cloud lifecycle | not run | — | Planned after Phase 3 |

## Deviations

- Planning only. No HIP code, build option, or documentation claim has shipped.
  V-1 through V-8 and M-1 through M-5 are specified but unexecuted; they must not
  be reported as passes.
- The cloud image versions in the planning input (ROCm 6.4.0 and CMake
  availability) are assumptions until M-1 records the actual values. If the
  image cannot build HIP or lacks CMake 3.21 or newer, the plan's Phase 1 gate
  fails and the packet is replanned before porting kernels.
- If the image lacks `rocprofv3`, V-6 is recorded as a gap rather than a pass;
  the correctness and timing checks still stand. Do not substitute a profiler
  that was not actually run.
- The repository already contains untracked or user-owned work at the time of
  planning. This packet does not modify kernel sources or existing build files.

## Definition of Done (Integration)

- [ ] Every automated and manual check executed, with evidence recorded
- [ ] Evidence traced to R-n requirements; deviations documented
- [ ] Specification, code, roadmap, and docs agree, and are merged as one
      coherent change
- [ ] Durable discoveries propagated to `tech-stack.md` and `mission.md`
