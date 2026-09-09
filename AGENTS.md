# Agent Instructions

This repository is a GPU kernel playground for CUDA and OpenCL experiments. The agentic workflow here is intentionally lightweight: use the repo docs as the operating contract, make changes that stay local to the requested kernel or docs task, and validate performance or correctness claims before presenting them as finished.

## Goal

Build, study, and improve standalone GPU kernels without breaking the backend split, build matrix, or correctness harnesses already present in this repo.

## Repo Surface

- `Readme.md` is the human-facing project overview.
- `CLAUDE.md` is the Claude Code-oriented repo guide.
- `docs/agentic-getting-started.md` is the main entrypoint for agentic use in this repo.
- `docs/cuda-agent-guide.md` and `docs/opencl-agent-guide.md` are the optimization rulebooks.
- `docs/opencl-3.1-migration-plan.md` and
  `docs/lora-linear-implementation-plan.md` track planned work; they are not
  implemented kernel claims.
- `.agents/rules/` contains mandatory shared rules for repo-local agents.
- `.agents/skills/` contains reusable custom-agent definitions.
- `.claude/agents/`, `.codex/agents/`, `.cursor/rules/`, `.github/agents/`, and `.opencode/agent/` project the same custom agents into each tool surface.
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
  `steady_clock` pair. Use `benchmarkWithReset()` when the output buffer is also an input.
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
reference implementation would otherwise dominate harness runtime.

`source/utils/benchmark_helpers.h` is separate and complementary: it converts a latency into
throughput, bandwidth, or a speedup ratio for harnesses that report those.

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

## Custom Agent Layout

- Shared rules: `.agents/rules/`
- Shared skills: `.agents/skills/`
- Claude agent entrypoints: `.claude/agents/`
- Codex agent entrypoints: `.codex/agents/`
- Cursor rules: `.cursor/rules/`
- GitHub Copilot agents: `.github/agents/`
- OpenCode agents: `.opencode/agent/`

The current repo-local custom agents are:

- `kernel-author`
- `cuda-optimizer`
- `opencl-reviewer`
- `perf-diagnoser`
- `docs-curator`
