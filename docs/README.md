# Documentation

> Audience: humans, agents · Applies to: all · Last reviewed: 2026-10

This directory describes how the repository works today: how to build and run
it, how to add kernels, how to optimize them, and how their cost is analyzed.
Planned or in-flight work lives separately in [`specs/`](../specs/) as dated
feature packages (`requirements.md`, `plan.md`, `validation.md`).

New to the repository? Start with the [project README](../Readme.md), then
[Adding a new kernel](guides/adding-a-new-kernel.md).

## Setup

- [GLSL compute backends](setup/glsl-backends.md) — headless OpenGL and Vulkan,
  shared shaders, correctness checks, and benchmark comparisons.

- [Running on AMD Developer Cloud](setup/amd-developer-cloud.md) — SSH, ROCm inspection,
  HIP build, correctness tests, benchmarks, and result collection.
- [Building on Google Colab](setup/google-colab.md) — build and run the
  C++/CUDA kernels on a free Colab GPU, or drive a runtime from the terminal
  with the optional Colab CLI.
- [External frameworks](setup/external-frameworks.md) — install CUTLASS, Mojo,
  Triton, and TinyGrad for comparison work.

## Guides

- [Adding a new kernel](guides/adding-a-new-kernel.md) — the CUDA and OpenCL
  kernel template, build wiring, and completion checklist.
- [Examples](guides/examples.md) — PyTorch, TinyGrad, Triton, and Mojo examples,
  with benchmarking notes.

## Optimization

- [CUDA optimization guide](optimization/cuda.md) — measurement protocol, hard
  rules, a priority-ordered playbook, and common mistakes.
- [OpenCL optimization guide](optimization/opencl.md) — host API, portability,
  work-group rules, and profiling.

## Complexity

- [Algorithmic complexity studies](complexity/README.md) — shared cost model,
  taxonomy, and kernel inventory.
- [GEMM study](complexity/gemm.md) — Phase 1 pilot.
- [Prefix sum study](complexity/prefix-sum.md) — Phase 1 pilot.
- [Sigmoid study](complexity/sigmoid.md) — Phase 1 pilot.
- [Phase 1 attempt ledger](complexity/phase-1-ledger.md) — the planner's record
  of the Phase 1 runs.

## References

- [References](references.md) — link-only notes (CUDA Rust) and pointers to the
  resource lists in the project README.

## Conventions

- File and directory names are lowercase kebab-case, except `README.md`; every
  document opens with `> Audience: ... · Applies to: ... · Last reviewed:
  YYYY-MM`.
- Every document under `docs/` is linked from this index.
- CI (`.github/workflows/docs.yml`) checks names, headers, index coverage, and
  links.
