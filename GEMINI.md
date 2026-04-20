# GEMINI.md

> `AGENTS.md` at the repo root is the single source of truth for repository conventions, agent layout, and workflow guidance.
> Keep this file aligned with `AGENTS.md` and prefer updating `AGENTS.md` first when repo rules change.

This file provides Gemini-oriented repository context for `gpu_playground`.

## Start Here

1. Read `AGENTS.md` first.
2. Use `Readme.md` for the human-facing repo overview.
3. Use `docs/agentic-getting-started.md` for the agentic entrypoint.
4. Use `docs/cuda-agent-guide.md` or `docs/opencl-agent-guide.md` depending on the backend.

## Repo Summary

- This repo contains standalone GPU kernel experiments in CUDA and OpenCL.
- Each kernel lives under `source/<kernel>/` with backend-specific subdirectories.
- The build is driven by `CMakeLists.txt` and `CMakePresets.json`.
- The shared repo-local agent surfaces live under `.agents`, `.claude`, `.codex`, `.cursor`, `.github`, and `.opencode`.

## Rule

If guidance here conflicts with `AGENTS.md`, follow `AGENTS.md`.
