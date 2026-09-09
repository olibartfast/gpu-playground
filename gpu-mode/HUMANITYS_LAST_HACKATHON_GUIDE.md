# GPU MODE — Humanity's Last Hackathon Guide

**Launches: May 4, 2026 | Platform: GPU MODE Leaderboard | Registration: https://huggingface.co/humanitys-last-hackathon**

> "This is not a normal hackathon. You will be judged on the context, not code!"

For toolchain setup (Popcorn CLI, Discord auth), see [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## Table of Contents
1. [Overview](#overview)
2. [Competition Format](#competition-format)
3. [Hardware Target](#hardware-target)
4. [Problems](#problems)
5. [Agent-Based Workflow](#agent-based-workflow)
6. [Creating a Submission](#creating-a-submission)
7. [Submitting to Leaderboard](#submitting-to-leaderboard)
8. [Workflow Summary](#workflow-summary)
9. [Optimization Strategies](#optimization-strategies)
10. [Learning Resources](#learning-resources)
11. [Quick Reference](#quick-reference)

---

## Overview

Humanity's Last Hackathon is a GPU MODE kernel competition focused on **AI model optimization for local inference**, targeting Apple Silicon (Mac Metal). The distinguishing feature is that participants are expected to use AI agents (Codex / Claude Code) to build and optimize kernels — the evaluation weighs *context and reasoning quality*, not just raw benchmark numbers.

- **Qualifiers**: Starts May 4, 2026
- **Finals**: Top performers on the leaderboard advance
- **Submission platform**: GPU MODE (Popcorn CLI)
- **Registration**: https://humanitys-last-hackathon-signup.hf.space

---

## Competition Format

- Submit kernel files via Popcorn CLI to the GPU MODE leaderboard
- Leaderboard-based ranking; top entries advance to the final battle
- Judged on **context** (reasoning quality, optimization approach) over raw code
- Problems will be published in `gpu-mode/reference-kernels` when the competition opens

---

## Hardware Target

**Primary:** Apple Silicon — Mac Metal (MLX / Metal Performance Shaders)

Unlike the AMD and NVIDIA tracks (which use Python/Triton/HIP), this competition targets:

| Concept | Mac Metal / MLX |
|---------|----------------|
| Compute framework | Metal Performance Shaders (MPS) / MLX |
| Kernel language | Metal Shading Language (MSL) or Python via MLX |
| Tensor cores | Apple Matrix Coprocessor (AMX) |
| Memory model | Unified Memory (CPU/GPU share DRAM) |

> **Note:** You do not need a Mac to develop. Submissions run remotely on Apple Silicon hardware
> via Popcorn CLI. Use `--mode dev` to test before official submissions.

---

## Problems

> Problem specifications will be published on May 4, 2026 in the GPU MODE reference-kernels repo.
> Update this section once the competition opens.

```bash
cd ~/gpu-mode/reference-kernels
git pull
ls problems/  # look for the humanitys-last or similar directory
```

Expected problem structure (based on prior competitions):
```
problems/<hackathon_dir>/
├── <problem>/
│   ├── reference.py   # PyTorch / MLX reference implementation
│   ├── submission.py  # Baseline to start from
│   ├── task.py        # Evaluation harness + custom_kernel signature
│   └── task.yml       # Shapes, constraints, scoring
```

---

## Agent-Based Workflow

This competition explicitly encourages using AI agents for optimization. The recommended loop:

1. **Read** the task spec (`task.yml`, `task.py`, `reference.py`)
2. **Profile** the baseline: identify the bottleneck (memory bandwidth? compute? latency?)
3. **Prompt your agent** with full context — the problem spec, the bottleneck, and your hypothesis
4. **Iterate**: one optimization per round, measure after each change
5. **Document** each step — the judging weighs *why* you made each decision

### Claude Code usage:
```bash
# From this repo root, start a session with the kernel as context
claude  # then describe the kernel task with reference.py pasted in
```

---

## Creating a Submission

```bash
mkdir -p ~/gpu-mode/submissions
cd ~/gpu-mode/submissions

# Copy the competition baseline
cp ~/gpu-mode/reference-kernels/problems/<hackathon_dir>/<problem>/submission.py \
   ./humanitys_last_<problem>.py

# Edit your kernel
vim humanitys_last_<problem>.py
```

Submission files must implement `custom_kernel(data)` matching the signature in `task.py`.

### Example skeleton (MLX / Metal):
```python
"""
Humanity's Last Hackathon — <problem_name>
Approach: <describe your optimization hypothesis here>
"""
from task import input_t, output_t
import mlx.core as mx


def custom_kernel(data: input_t) -> output_t:
    # TODO: replace with optimized implementation
    raise NotImplementedError("Fill in your kernel here")
```

Starter submissions are in `gpu-mode/submissions/` — check this directory after May 4.

---

## Submitting to Leaderboard

### Test run (dev mode — does not affect ranking):
```bash
cd ~/gpu-mode
./tools/submit.sh -g <GPU_MODEL> -l <LEADERBOARD_NAME> --mode dev \
  submissions/humanitys_last_<problem>.py
```

### Official submission:
```bash
cd ~/gpu-mode
./tools/submit.sh -g <GPU_MODEL> -l <LEADERBOARD_NAME> \
  submissions/humanitys_last_<problem>.py
```

### Browse leaderboards:
```bash
~/gpu-mode/popcorn-cli/target/release/popcorn-cli submit
```

> **GPU model and leaderboard names**: check Discord announcements on May 4 for the exact values.

### Check standings (Discord):
```
/leaderboard <leaderboard_name>
```

---

## Workflow Summary

```bash
# 1. One-time setup (if not already done)
./tools/setup.sh
export POPCORN_API_URL="<url_from_discord>"

# 2. Pull problems on May 4
cd ~/gpu-mode/reference-kernels && git pull

# 3. Create submission
cp problems/<dir>/<problem>/submission.py ~/gpu-mode/submissions/humanitys_last_<problem>.py
vim ~/gpu-mode/submissions/humanitys_last_<problem>.py

# 4. Test
./tools/submit.sh -g <GPU> -l <LEADERBOARD> --mode dev submissions/humanitys_last_<problem>.py

# 5. Submit
./tools/submit.sh -g <GPU> -l <LEADERBOARD> submissions/humanitys_last_<problem>.py

# 6. Check
# Discord: /leaderboard <leaderboard_name>
# Web: https://www.gpumode.com
```

---

## Optimization Strategies

### Level 1: Baseline (MLX built-ins)
Use standard MLX operations that dispatch to Metal Performance Shaders internally:
```python
import mlx.core as mx
result = mx.matmul(A, B)
```
- Correct and easy
- Not competitive (cuBLAS/rocBLAS equivalent)

### Level 2: MLX-Optimized
- Avoid unnecessary copies and type conversions
- Use `mx.compile()` for kernel fusion
- Operate entirely on the Metal device (no numpy round-trips)
- Use `mx.stream()` for asynchronous execution

### Level 3: Custom Metal Kernel via MLX
MLX supports embedding raw Metal Shading Language kernels:
```python
import mlx.core as mx

source = """
kernel void my_kernel(
    device const float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    C[index] = A[index] * 2.0f;
}
"""
kernel = mx.fast.metal_kernel(
    name="my_kernel",
    input_names=["A"],
    output_names=["C"],
    source=source,
)
```

### Level 4: Tiled Metal Kernel with Threadgroup Memory
- Use `threadgroup` memory (equivalent to shared memory / LDS)
- Tile matrix dimensions to match Apple Matrix Coprocessor block sizes
- Use `simdgroup_matrix` / `simdgroup_multiply_accumulate` for tensor-core-style ops
- Tune `threads_per_threadgroup` to match the A-series/M-series GPU topology

---

## Learning Resources

### Apple / Metal:
- **MLX documentation**: https://ml-explore.github.io/mlx/
- **MLX custom Metal kernels**: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- **Metal Performance Shaders**: https://developer.apple.com/documentation/metalperformanceshaders
- **Metal Shading Language Spec**: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- **simdgroup_matrix reference**: search Apple developer docs for `simdgroup_multiply_accumulate`

### GPU MODE:
- **Discord**: https://discord.gg/gpumode
- **YouTube**: https://www.youtube.com/@GPUMODE
- **Website / leaderboards**: https://www.gpumode.com
- **Reference kernels**: https://github.com/gpu-mode/reference-kernels
- **Popcorn CLI**: https://github.com/gpu-mode/popcorn-cli

---

## Quick Reference

```bash
# Authenticate (once)
export POPCORN_API_URL="<url_from_discord>"
~/gpu-mode/popcorn-cli/target/release/popcorn-cli register discord

# Browse leaderboards
~/gpu-mode/popcorn-cli/target/release/popcorn-cli submit

# Dev test
./tools/submit.sh -g <GPU> -l <LEADERBOARD> --mode dev submissions/humanitys_last_<problem>.py

# Official submission
./tools/submit.sh -g <GPU> -l <LEADERBOARD> submissions/humanitys_last_<problem>.py

# Check Discord
/leaderboard <leaderboard_name>
```

**Resources:**
- GPU MODE Website: https://www.gpumode.com
- GPU MODE YouTube: https://www.youtube.com/@GPUMODE
- Reference Kernels: https://github.com/gpu-mode/reference-kernels
- Popcorn CLI: https://github.com/gpu-mode/popcorn-cli
- MLX Docs: https://ml-explore.github.io/mlx/
- Registration: https://humanitys-last-hackathon-signup.hf.space

---

**Questions?** Ask in the GPU MODE Discord: https://discord.gg/gpumode
