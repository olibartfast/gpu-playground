# Codex Guide

This repo does not currently ship a checked-in `.codex/` configuration tree. Codex can still work against the same repo contract and docs used by Claude or other coding agents.

## Canonical Docs

- Repo rules: [`../AGENTS.md`](../AGENTS.md)
- Project overview: [`../Readme.md`](../Readme.md)
- Claude-native repo summary: [`../CLAUDE.md`](../CLAUDE.md)
- CUDA rules: [`cuda-agent-guide.md`](cuda-agent-guide.md)
- OpenCL rules: [`opencl-agent-guide.md`](opencl-agent-guide.md)

## Recommended Workflow

1. Start Codex in the repo root.
2. Read `AGENTS.md` first.
3. Inspect the target kernel and the matching backend guide.
4. Keep the task scoped to one kernel, one backend, or one docs slice unless the request explicitly asks for a larger refactor.
5. Build and run the relevant validation path before closing the task.

## Scope Notes

- Treat `Readme.md`, `AGENTS.md`, and `docs/` as the source of truth for repo workflows.
- Do not invent repo-local Codex config or agent files unless the task explicitly asks for them.
- If the repo later adds `.codex/` assets, update this guide to point at them directly.
