# Documentation

> Audience: humans, agents · Applies to: all · Last reviewed: 2026-10

This directory describes how the repository works today. Planned or in-flight
changes live in [`specs/`](../specs/) as dated feature packages
(`requirements.md`, `plan.md`, `validation.md`).

## Setup

- [Building on Google Colab](setup/google-colab.md) — build and run the C++/CUDA kernels on a free Colab GPU.
- [External frameworks](setup/external-frameworks.md) — install CUTLASS, Mojo, Triton, and TinyGrad for comparison work.

## Guides

- [Adding a new kernel](guides/adding-a-new-kernel.md) — the CUDA + OpenCL kernel template, build wiring, and checklist.
- [Examples](guides/examples.md) — PyTorch, TinyGrad, Triton, and Mojo GPU examples plus benchmarking notes.

## Optimization

- [CUDA optimization guide](optimization/cuda.md) — measurement protocol, hard rules, priority-ordered playbook, common mistakes.
- [OpenCL optimization guide](optimization/opencl.md) — host API, portability, work-group rules, and profiling.

## Complexity

- [Algorithmic complexity studies](complexity/README.md) — shared cost model, improvement taxonomy, and kernel inventory.
- [GEMM study](complexity/gemm.md) — Phase 1 pilot.
- [Prefix sum study](complexity/prefix-sum.md) — Phase 1 pilot.
- [Sigmoid study](complexity/sigmoid.md) — Phase 1 pilot.
- [Phase 1 attempt ledger](complexity/phase-1-ledger.md) — planner record of the Phase 1 runs.

## References

- [References](references.md) — link-only notes (CUDA Rust) and pointers to the Readme resource lists.

## Conventions

- File and directory names are lowercase kebab-case; `README.md` is the only uppercase name.
- Every doc starts with a header block:
  `> Audience: ... · Applies to: ... · Last reviewed: YYYY-MM`.
- Every doc under `docs/` is linked from this index.
- CI (`.github/workflows/docs.yml`) checks names, headers, index coverage, and links.
