# Agentic Getting Started

This repo now treats agentic documentation as a small, curated set instead of loose notes. Start here, then branch into the tool-specific and backend-specific guides.

## What This Doc Set Covers

- A repo-wide operating contract in [`../AGENTS.md`](../AGENTS.md)
- Claude Code usage in [`claude-code-guide.md`](claude-code-guide.md)
- Codex usage in [`codex-guide.md`](codex-guide.md)
- CUDA optimization rules in [`cuda-agent-guide.md`](cuda-agent-guide.md)
- OpenCL optimization rules in [`opencl-agent-guide.md`](opencl-agent-guide.md)

## First Session Checklist

1. Read [`../Readme.md`](../Readme.md) for the project layout and build options.
2. Read [`../AGENTS.md`](../AGENTS.md) for repo rules.
3. Pick the backend guide that matches your task:
   - CUDA: [`cuda-agent-guide.md`](cuda-agent-guide.md)
   - OpenCL: [`opencl-agent-guide.md`](opencl-agent-guide.md)
4. Build the repo using the backend you plan to touch.
5. Run the target kernel before editing it.

## Recommended Session Pattern

1. Start from one concrete task:
   - optimize one kernel
   - add one kernel
   - fix one backend bug
   - improve one doc cluster
2. Read the target directory under `source/<kernel>/`.
3. Inspect the existing harness in `main.cpp` before changing implementation files.
4. Make one attributable change at a time.
5. Re-run the relevant build and executable.
6. Record whether the outcome is correctness-only, performance-only, or both.

## Repo Entry Points

- Human overview: [`../Readme.md`](../Readme.md)
- Claude-native repo guidance: [`../CLAUDE.md`](../CLAUDE.md)
- Agent operating contract: [`../AGENTS.md`](../AGENTS.md)
- Kernel authoring guide: [`adding-a-new-kernel.md`](adding-a-new-kernel.md)
- Colab path: [`building-on-google-colab.md`](building-on-google-colab.md)

## Agent Folder Layout

The repo-local agent surfaces now follow the same pattern used by the reference repos:

- shared rules: `.agents/rules/`
- shared skills: `.agents/skills/`
- Claude agent entrypoints: `.claude/agents/`
- Codex agent entrypoints: `.codex/agents/`
- Cursor agent rules: `.cursor/rules/`
- GitHub Copilot agent entrypoints: `.github/agents/`
- OpenCode agent entrypoints: `.opencode/agent/`
- OpenCode repo config: `opencode.json`

## What Changed From The Old Layout

The previous `docs/agentic_ai/` folder mixed CUDA rules, OpenCL guidance, and agent-facing notes without a clear entrypoint. The new structure makes the contract explicit:

- one repo-level instructions file
- one getting-started document
- one guide per tool surface
- one guide per backend optimization surface
