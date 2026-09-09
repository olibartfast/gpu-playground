# Claude Code Guide

Claude Code is the only checked-in agent-specific integration already present in this repo today.

## Repo Surface

- Repo rules live in [`../AGENTS.md`](../AGENTS.md).
- Claude-specific repo context lives in [`../CLAUDE.md`](../CLAUDE.md).
- Local Claude settings currently live in `.claude/settings.local.json`.

## Recommended Workflow

1. Start Claude Code in the repo root.
2. Read `AGENTS.md`, then `CLAUDE.md`.
3. Read the backend guide that matches the task.
4. Inspect the relevant kernel directory before proposing edits.
5. Build and run the smallest validation path that proves the change.

## Good Task Shapes

- "Profile `source/gemm` and explain the bottleneck."
- "Optimize the CUDA version of `softmax` without changing the OpenCL path."
- "Add a new kernel following the existing backend split."
- "Refactor docs and keep `Readme.md`, `CLAUDE.md`, and `docs/` in sync."

## Expectations

- Keep changes local to the requested surface.
- Preserve the current CMake and backend organization.
- When changing CUDA or OpenCL code, use the corresponding backend guide as the default review checklist.
- If a workflow change affects Claude usage, update both `CLAUDE.md` and this guide.
