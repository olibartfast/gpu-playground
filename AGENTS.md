# Agent Instructions

This repository is a GPU kernel playground for CUDA and OpenCL experiments. The agentic workflow here is intentionally lightweight: use the repo docs as the operating contract, make changes that stay local to the requested kernel or docs task, and validate performance or correctness claims before presenting them as finished.

## Goal

Build, study, and improve standalone GPU kernels without breaking the backend split, build matrix, or correctness harnesses already present in this repo.

## Repo Surface

- `Readme.md` is the human-facing project overview.
- `docs/agentic-getting-started.md` is the main entrypoint for agentic use in this repo.
- `docs/cuda-agent-guide.md` and `docs/opencl-agent-guide.md` are the optimization rulebooks.
- `docs/opencl-3.1-migration-plan.md` and
  `docs/lora-linear-implementation-plan.md` track planned work; they are not
  implemented kernel claims.
- This file is the single source of truth for agent rules and agent roles (see
  "Agent Rules" and "Agent Roles" below). There are no per-tool agent trees.
- `opencode.json` is the repo-local OpenCode config entrypoint.
- `source/<kernel>/` contains the implementation for each kernel.
- `source/utils/` contains shared CUDA and OpenCL helper code.

## Hard Rules

- Keep changes scoped to the task. Do not refactor unrelated kernels while working on one kernel.
- Preserve the existing backend split:
  - CUDA code in `source/<kernel>/cuda/`
  - OpenCL C API code in `source/<kernel>/opencl/`
  - OpenCL C++ wrapper code in `source/<kernel>/opencl_cpp/`
- Keep public headers backend-agnostic where the repo already follows that pattern.
- Use `CUDA_CHECK` and `CL_CHECK` style error handling consistently.
- Do not claim a CUDA or OpenCL optimization without either:
  - a correctness check, or
  - a clear note that validation was not run.
- Prefer small, attributable optimization changes over multi-variable rewrites.
- Time every CPU/GPU path with `benchmark()` from `source/utils/benchmark.h`, never a single
  `steady_clock` pair. Use `benchmarkWithReset()` when the output buffer is also an input. Use
  `benchmarkDevice()` for kernel-only time (e.g. CUDA events) alongside end-to-end.
- Every new kernel must update `Readme.md` in the same change. At minimum,
  update the kernel inventory, backend coverage, and any required build or run
  instructions.
- Update docs when workflow or repo structure changes.

## Build And Run

Default CUDA path:

```bash
cmake --preset default
cmake --build --preset default -j$(nproc)
```

OpenCL C API:

```bash
cmake -B build/opencl -DUSE_OPENCL=ON
cmake --build build/opencl -j$(nproc)
```

OpenCL C++ wrapper:

```bash
cmake -B build/opencl_cpp -DUSE_OPENCL_CPP=ON
cmake --build build/opencl_cpp -j$(nproc)
```

Run one kernel:

```bash
./build/default/source/gemm/gemm
```

Profile a CUDA binary:

```bash
./cuda_perf_analysis.sh ./build/default/source/gemm/gemm
```

## Testing And Benchmarking

Each kernel binary is its own test harness; there is no separate unit-test framework. A harness
validates GPU output against a CPU reference within a stated tolerance (~1e-4f) and reports timings.

Time with `benchmark()` / `benchmarkWithReset()` from `source/utils/benchmark.h` (backend-agnostic,
pulls in no CUDA or OpenCL headers). A single timed call mostly measures one-off costs — CUDA context
creation, `clBuildProgram`, first-touch page faults, clock ramp-up — which can dwarf the kernel itself.
The protocol is: a few untimed warm-up calls, then repeated timed iterations reported as median plus
spread, compared on the median.

```cpp
#include "benchmark.h"

BenchResult cpu_bench = benchmark([&] { softmax_cpu(input, out_cpu, N); });
BenchResult gpu_bench = benchmark([&] { softmax_gpu(input, out_gpu, N); });
printBench("CPU:", cpu_bench);
printBench("GPU:", gpu_bench);
printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);

// Output doubles as input: reset before every call, untimed.
BenchResult gemm_bench = benchmarkWithReset(
    [&] { gemmGPU(A, B, C, alpha, beta, M, N, K); },
    [&] { std::copy(C_seed, C_seed + M * N, C); });
```

Defaults are 3 warm-up + 10 timed iterations; pass explicit counts as trailing arguments when a slow
reference implementation would otherwise dominate harness runtime. `summarize()` turns
already-collected samples (warm-up excluded) into a `BenchResult` directly, for harnesses that
gather timings outside the `benchmark()`/`benchmarkWithReset()` loop.

For kernel-only time separate from host overhead, use `benchmarkDevice()` /
`benchmarkDeviceWithReset()`: `fn` returns that call's device-measured milliseconds (e.g. CUDA
events via a `float* kernel_time_ms` out-parameter), and the result's `end_to_end` and `device`
fields are each a `BenchResult`. Sync contract: a timed callable must still block until its device
work completes (a blocking D2H copy counts), otherwise only launch overhead is measured.

```cpp
// my_kernel_gpu stands for any host wrapper exposing a float* kernel_time_ms out-parameter.
DeviceBenchResult gpu = benchmarkDevice([&] {
    float kernel_ms = 0.0f;
    out = my_kernel_gpu(a, b, n, &kernel_ms);
    return kernel_ms;
});
printBench("GPU end-to-end:", gpu.end_to_end);
printBench("GPU kernel:", gpu.device);
```

`source/utils/benchmark_helpers.h` is separate and complementary: it holds unit converters
(`gpu_benchmark::giga_operations_per_second`, `gigabytes_per_second`, `million_items_per_second`,
`speedup`) applied to a result's `median_ms` to report throughput, bandwidth, or a speedup ratio.

## Standard Workflow

1. Read `Readme.md`, this file, and the relevant backend guide.
2. Inspect the target kernel's `main.cpp` plus the backend-specific implementation files.
3. Build a correctness baseline before tuning.
4. Change one optimization hypothesis at a time.
5. Rebuild, rerun, and record what changed.
6. For a new kernel, update `Readme.md` before presenting the work as complete.
7. If the task changes the workflow or conventions, update the matching docs.

## Which Guide To Read

- Starting an agentic session in this repo: `docs/agentic-getting-started.md`
- Working from Claude Code: `docs/claude-code-guide.md`
- Working from Codex: `docs/codex-guide.md`
- CUDA optimization tasks: `docs/cuda-agent-guide.md`
- OpenCL optimization tasks: `docs/opencl-agent-guide.md`
- OpenCL 3.1 migration work: `docs/opencl-3.1-migration-plan.md`
- Planned LoRA linear work: `docs/lora-linear-implementation-plan.md`

## GPU MODE Competitions

| Competition | Guide | Status | Hardware |
|-------------|-------|--------|----------|
| General setup | `gpu-mode/SETUP_GUIDE.md` | Ongoing | NVIDIA (B200) |
| AMD Hackathon | `gpu-mode/AMD_HACKATHON_GUIDE.md` | Phase 2 (Finals) | AMD MI355X |
| Humanity's Last Hackathon | `gpu-mode/HUMANITYS_LAST_HACKATHON_GUIDE.md` | Opens May 4, 2026 | Apple Silicon (Metal) |

Submissions live in `gpu-mode/submissions/`. Tools in `gpu-mode/tools/`.

## Agent Rules

These rules are mandatory for every agent and every tool (Claude Code, Codex,
Cursor, Copilot, OpenCode). They complement the Hard Rules above.

### Backend Boundaries

- Keep `main.cpp` as the backend-agnostic harness entrypoint.
- Do not leak CUDA or OpenCL types into backend-agnostic headers unless the existing file already does.
- Keep backend plumbing (includes, resource lifetimes) in the backend implementation file, not the harness.
- Check: would another backend still compile cleanly after this change?

### Kernel Harness Contract

- `main.cpp` drives the CPU reference and the GPU execution; allocation, transfers, and launches live
  in backend implementation files.
- Return non-zero on validation failure.
- Keep the harness focused on setup, invocation, timing, and comparison.
- Keep naming and directory structure aligned with existing kernels.

### CUDA Optimization

- Start from a baseline; change one optimization variable at a time.
- Use thread counts that are multiples of `32`; treat `128-256` threads per block as a starting point, not a law.
- Prioritize coalesced memory access before micro-optimizations.
- Use shared memory only when reuse clearly offsets synchronization and storage cost.
- Wrap CUDA runtime calls with `CUDA_CHECK`; check launches with `cudaGetLastError()`.
- Avoid device-side allocation. Make precision and fast-math tradeoffs explicit.

### OpenCL Portability

- Query optional device capabilities instead of assuming them; guard optional OpenCL C features
  with `__opencl_c_*` checks.
- Keep fallback paths clear when sub-groups, FP16, or other optional features are absent.
- Prefer simple coalesced access before adding `__local` tiling.
- Re-validate local-size choices on the actual target device class.
- Always inspect and surface build logs when kernel compilation fails.

### Profiling And Validation

- Do not claim a performance win without naming the measurement path.
- Distinguish clearly between correctness verified, build verified, and performance measured.
- Prefer the smallest validation step that proves the requested change.
- For CUDA performance work, start with `./cuda_perf_analysis.sh <binary>` when appropriate.
- When full validation is not possible, state the gap explicitly.

## Agent Roles

Any tool can adopt one of these roles when a task matches. Each role follows all Agent Rules above.

| Role | Use when | Report |
|------|----------|--------|
| `kernel-author` | Adding or restructuring a kernel; wiring its `CMakeLists.txt` and harness | Kernel surface changed, backend dirs affected, build/harness implications, `Readme.md` update, validation status |
| `cuda-optimizer` | A CUDA kernel is slow or needs tuning/review | Bottleneck hypothesis, evidence, highest-value next change, validation status and risks |
| `opencl-reviewer` | OpenCL kernel or host path needs review; portability, local-size, or optional-feature concerns | Issue identified, evidence, recommended next step, validation status and compatibility risks |
| `perf-diagnoser` | A kernel regressed, a profile needs interpreting, or the optimization direction is unclear | Bottleneck class (harness overhead, CUDA memory, CUDA occupancy/divergence, OpenCL portability/local-size, algorithmic work inflation, measurement gap), evidence, next step, tradeoffs |
| `docs-curator` | Docs, entrypoints, or tool guides drift from repo structure | Entrypoints updated, structure described, stale references removed, validation status |

Across roles: prefer small changes that fit the existing template, no broad rewrites without a
measured or clearly argued bottleneck, and one source of truth over duplicated prose.
