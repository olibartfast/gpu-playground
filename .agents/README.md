# Custom Agents

This directory holds reusable custom-agent definitions for the GPU Playground repository.

Use one directory per agent under `.agents/skills/`. Each agent directory should contain:

- `agent.md`: purpose, scope, expected inputs and outputs, and constraints
- `skill.md`: reusable instructions to give that agent

Shared repo rules live in `.agents/rules/` and are mandatory for all custom agents.

Current starter agents:

- `kernel-author`
- `cuda-optimizer`
- `opencl-reviewer`
- `perf-diagnoser`
- `docs-curator`
