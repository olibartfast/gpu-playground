# GPU MODE AMD Hackathon Setup Guide

**AMD Instinct MI355X — Qualifiers: March 6-April 6, 2026**

This guide covers how to set up your environment and submit kernels for the GPU MODE AMD Hackathon,
sponsored by AMD. For the NVIDIA track, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Table of Contents
1. [Overview & Timeline](#overview--timeline)
2. [Prerequisites](#prerequisites)
3. [Step 1: Install Rust and Cargo](#step-1-install-rust-and-cargo)
4. [Step 2: Install Popcorn CLI](#step-2-install-popcorn-cli)
5. [Step 3: Join GPU MODE Discord](#step-3-join-gpu-mode-discord)
6. [Step 4: Authenticate Popcorn CLI](#step-4-authenticate-popcorn-cli)
7. [Step 5: Clone Reference Kernels](#step-5-clone-reference-kernels)
8. [Step 6: Understand the Phase 1 Problems](#step-6-understand-the-phase-1-problems)
9. [Step 7: Create Your First Submission](#step-7-create-your-first-submission)
10. [Step 8: Submit to Leaderboard](#step-8-submit-to-leaderboard)
11. [Step 9: Check Your Position](#step-9-check-your-position)
12. [Workflow Summary](#workflow-summary)
13. [Optimization Strategies](#optimization-strategies)
14. [Learning Resources](#learning-resources)
15. [Troubleshooting](#troubleshooting)
16. [Quick Reference Commands](#quick-reference-commands)

---

## Overview & Timeline

**Phase 1: Qualifiers** — March 6 – April 6, 2026
- Optimize three GPU kernels targeting AMD Instinct MI355X

**Phase 2: Finals** — April 7 – May 15, 2026
- End-to-end LLM inference optimization (DeepSeek-R1, Kimi K2.5)


**Phase 1 Kernels:**
| Kernel | Description |
|--------|-------------|
| MXFP4 GEMM | Matrix multiplication with MXFP4 block scaling |
| MLA Decode | Multi-head Latent Attention decode (DeepSeek-style) |
| MXFP4 MoE | Mixture-of-Experts routing + GEMM with MXFP4 |

---

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or WSL2 on Windows
- **Python**: 3.8 or higher
- **Rust**: Latest stable version (for building Popcorn CLI)
- **Git**: For cloning repositories
- **ROCm** (optional): Only needed if you have a local AMD GPU for testing

### Accounts Required
1. **Discord Account** — Join GPU MODE Discord: https://discord.gg/gpumode
2. **GitHub Account** (optional) — Alternative authentication method

> **Note:** You do NOT need a local AMD GPU. Submissions run remotely on MI355X hardware
> via the Popcorn CLI. Use `--mode dev` to test before official submissions.

---

## Step 1: Install Rust and Cargo

Rust is required to build the Popcorn CLI submission tool.

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the prompts, then source the environment
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

---

## Step 2: Install Popcorn CLI

Popcorn CLI is the submission tool used to send kernels to remote AMD hardware.

```bash
export WORKSPACE=gpu-mode
mkdir -p ~/$WORKSPACE
cd ~/$WORKSPACE

# Clone and build Popcorn CLI
git clone https://github.com/gpu-mode/popcorn-cli.git
cd popcorn-cli
./build.sh

# The binary will be at: target/release/popcorn-cli
```

### Add to PATH:
```bash
sudo cp target/release/popcorn-cli /usr/local/bin/
# or
echo 'alias popcorn="~/$WORKSPACE/popcorn-cli/target/release/popcorn-cli"' >> ~/.bashrc
source ~/.bashrc
```

### Verify:
```bash
popcorn-cli --help
```

---

## Step 3: Join GPU MODE Discord

1. Go to https://discord.gg/gpumode and join the server
2. Find the AMD competition channel (look for `#amd-competition` or similar in pinned messages)
3. Read pinned messages for leaderboard names and announcements

---

## Step 4: Authenticate Popcorn CLI

### Get API URL from Discord:
In any Discord channel, type:
```
/get-api-url
```
The bot responds with your API URL.

### Set Environment Variable:
```bash
export POPCORN_API_URL="<url_from_discord>"

# Make persistent
echo 'export POPCORN_API_URL="<url_from_discord>"' >> ~/.bashrc
source ~/.bashrc
```

### Register:
```bash
cd ~/$WORKSPACE/popcorn-cli
./target/release/popcorn-cli register discord
```

This opens a browser for Discord OAuth and saves your CLI ID to `~/.popcorn.yaml`.

### Verify:
```bash
cat ~/.popcorn.yaml
# Should show your cli_id
```

---

## Step 5: Clone Reference Kernels

```bash
cd ~/$WORKSPACE
git clone https://github.com/gpu-mode/reference-kernels.git
cd reference-kernels

# List available problem directories
ls problems/
# Find the current AMD qualifier directory (currently problems/amd_202602)
cd problems/amd_202602
ls
# Current qualifier problems: mxfp4-mm/, mixed-mla/, moe-mxfp4/
```

### Explore a Problem:
```bash
cd mxfp4-mm
ls
# - reference.py   : Reference implementation
# - submission.py  : Baseline kernel to start from
# - task.yml       : Problem specification (shapes, constraints)
# - task.py        : Task runner / evaluation harness
```

---

## Step 6: Understand the Phase 1 Problems

### Problem 1: MXFP4 GEMM

**What to implement:**
```python
C = A @ B.T  # Matrix multiplication with MXFP4 block scaling
```

**Key concepts:**
- **MXFP4**: 4-bit floating point with shared block scale factors (every 16–32 elements share one scale)
- **Target**: AMD MI355X Matrix Cores (MFMA instructions)
- **Goal**: Maximize MXFP4 GEMM throughput using Matrix Core acceleration

---

### Problem 2: MLA Decode

**What to implement:**
Multi-head Latent Attention decode step (as used in DeepSeek-R1).

**Key concepts:**
- **MLA**: Compresses the KV cache into a latent vector; requires an up-projection at decode time
- Each decode step attends over the compressed KV cache
- **Goal**: Minimize latency for single-token decode on MI355X

---

### Problem 3: MXFP4 MoE

**What to implement:**
Mixture-of-Experts forward pass: token routing + batched expert GEMMs with MXFP4.

**Key concepts:**
- **MoE routing**: Each token is dispatched to K of N experts
- **Expert GEMMs**: Irregular batched matrix multiplications (variable token counts per expert)
- **Goal**: Maximize throughput on MI355X with MXFP4 expert weights

---

## Step 7: Create Your First Submission

```bash
mkdir -p ~/$WORKSPACE/submissions
cd ~/$WORKSPACE/submissions

# Copy a template to start from
cp ~/$WORKSPACE/reference-kernels/problems/amd_202602/mxfp4-mm/submission.py \
   ./my_mxfp4_mm.py

# Edit your kernel
vim my_mxfp4_mm.py
```

Submissions are Python files. You can use:
- **PyTorch-ROCm** — easiest starting point
- **Triton** — Python kernel language with built-in ROCm backend
- **HIP C++ extension** — for maximum performance (compile via `torch.utils.cpp_extension`)

The submission file must implement a `custom_kernel(data)` function matching the signature in `task.py`.

---

## Step 8: Submit to Leaderboard

### Browse Available Leaderboards:
```bash
cd ~/$WORKSPACE/popcorn-cli
./target/release/popcorn-cli submit
# Opens interactive TUI to browse leaderboards
```

### Test Run (dev mode — does not count on leaderboard):
```bash
./target/release/popcorn-cli submit \
  --gpu MI355X \
  --leaderboard mxfp4-mm \
  --mode dev \
  ~/$WORKSPACE/submissions/my_mxfp4_mm.py
```

### Official Submission:
```bash
./target/release/popcorn-cli submit \
  --gpu MI355X \
  --leaderboard mxfp4-mm \
  --mode leaderboard \
  ~/$WORKSPACE/submissions/my_mxfp4_mm.py
```

Repeat for the other two leaderboards (`mla-py`, `3_moe_mxfp4`). Verify the exact leaderboard
names via the TUI or Discord.

---

## Step 9: Check Your Position

### Discord:
```
/leaderboard mxfp4-mm
/leaderboard mla-py
/leaderboard 3_moe_mxfp4
```

### Web:
Visit https://www.gpumode.com for live standings.

---

## Workflow Summary

```bash
# 1. Set up once
export POPCORN_API_URL="your_api_url_from_discord"

# 2. Write your kernel
cd ~/$WORKSPACE/submissions
vim my_kernel.py

# 3. Test remotely (dev mode)
cd ~/$WORKSPACE/popcorn-cli
./target/release/popcorn-cli submit \
  --gpu MI355X \
  --leaderboard mxfp4-mm \
  --mode dev \
  ~/$WORKSPACE/submissions/my_kernel.py

# 4. Submit officially
./target/release/popcorn-cli submit \
  --gpu MI355X \
  --leaderboard mxfp4-mm \
  --mode leaderboard \
  ~/$WORKSPACE/submissions/my_kernel.py

# 5. Check results in Discord
# /leaderboard mxfp4-mm
```

---

## Optimization Strategies

### Level 1: Baseline (PyTorch-ROCm)
Use standard PyTorch ops that map to rocBLAS/MIOpen internally:
```python
C = torch.mm(A, B.T)
```
- Easy to implement and correct
- Far from peak MI355X performance

### Level 2: GPU-Optimized PyTorch-ROCm
- Remove all `.cpu()` / `.numpy()` round-trips
- Use pre-permuted scale factors
- Compile with `@torch.compile` (uses ROCm backend)
- Batch operations to hide kernel launch overhead

### Level 3: Triton Kernel
Triton has a built-in ROCm backend — the same Python syntax targets AMDGPU:
```python
import triton
import triton.language as tl

@triton.jit
def mxfp4_gemm_kernel(A, B, C, ...):
    a = tl.load(A + ...)
    b = tl.load(B + ...)
    acc = tl.dot(a, b)      # Maps to MFMA on AMD
    tl.store(C + ..., acc)
```
- Manual tiling and blocking for MI355X cache hierarchy
- Competitive performance without writing HIP C++

### Level 4: HIP C++ with MFMA / rocWMMA
Write native AMD GPU kernels for maximum throughput:

**Key AMD-specific concepts:**
| Concept | AMD | CUDA Equivalent |
|---------|-----|-----------------|
| Thread group | Wavefront (64 threads) | Warp (32 threads) |
| Tensor core | MFMA (Matrix Fused Multiply-Add) | Tensor Core / WMMA |
| Shared memory | LDS (Local Data Share) | Shared memory |
| High-level MFMA wrapper | rocWMMA | WMMA / CuTe |
| Batched GEMM library | hipBLASLt | cuBLASLt |

**MFMA intrinsic example:**
```cpp
// 16x16x16 FP16 matrix multiply-accumulate
__builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, c_frag, 0, 0, 0);
```

**rocWMMA example (higher-level):**
```cpp
#include <rocwmma/rocwmma.hpp>
rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half> a_frag;
rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag;
rocwmma::load_matrix_sync(a_frag, A_ptr, lda);
rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

---

## Learning Resources

### AMD / ROCm:
- **ROCm Documentation**: https://rocm.docs.amd.com
- **HIP Programming Guide**: https://rocm.docs.amd.com/projects/HIP/
- **rocWMMA** (MFMA C++ wrappers): https://github.com/ROCm/rocWMMA
- **hipBLASLt** (batched/strided GEMM): https://github.com/ROCm/hipBLASLt
- **AMD Instinct MI300X ISA Reference**: search AMD developer docs for CDNA3 ISA

### Triton (ROCm backend):
- **Triton repo**: https://github.com/triton-lang/triton (ROCm backend included)
- Triton tutorials work on AMD — just install the ROCm wheel

### GPU MODE Community:
- **Discord**: https://discord.gg/gpumode — AMD competition channel, `#general`
- **YouTube**: https://www.youtube.com/@GPUMODE — kernel optimization lectures
- **Website**: https://www.gpumode.com — leaderboards and announcements

### Reference Kernels:
- https://github.com/gpu-mode/reference-kernels — baseline implementations and task specs

---

## Troubleshooting

### Authentication Issues (401 Unauthorized):
```bash
rm ~/.popcorn.yaml
cd ~/$WORKSPACE/popcorn-cli
./target/release/popcorn-cli register discord
cat ~/.popcorn.yaml  # Verify cli_id present
```

### Popcorn CLI Build Errors:
```bash
rustup update stable
cd ~/$WORKSPACE/popcorn-cli
cargo clean
./build.sh
```

### Can't Find AMD Leaderboard:
- Use the TUI: `./target/release/popcorn-cli submit` (no args) to browse all leaderboards
- Ask in Discord: `/leaderboard` to list all available
- The exact names may differ from what is documented here — check Discord announcements

### Submission Validation Errors:
- Check `task.py` in the problem folder for the exact `custom_kernel` signature
- Test with `--mode dev` before `--mode leaderboard`
- Ask in the AMD competition channel on Discord

---

## Quick Reference Commands

```bash
# Authenticate (once)
export POPCORN_API_URL="<url_from_discord>"
~/$WORKSPACE/popcorn-cli/target/release/popcorn-cli register discord

# Browse leaderboards
~/$WORKSPACE/popcorn-cli/target/release/popcorn-cli submit

# Test submission (dev mode)
~/$WORKSPACE/popcorn-cli/target/release/popcorn-cli submit \
  --gpu MI355X \
  --leaderboard mxfp4-mm \
  --mode dev \
  ~/$WORKSPACE/submissions/my_kernel.py

# Official submission
~/$WORKSPACE/popcorn-cli/target/release/popcorn-cli submit \
  --gpu MI355X \
  --leaderboard mxfp4-mm \
  --mode leaderboard \
  ~/$WORKSPACE/submissions/my_kernel.py

# Check standings (Discord)
/leaderboard mxfp4-mm
/leaderboard mla-py
/leaderboard 3_moe_mxfp4
```

---

**Questions?** Ask in the AMD competition channel on GPU MODE Discord: https://discord.gg/gpumode

**Resources:**
- GPU MODE Website: https://www.gpumode.com
- GPU MODE YouTube: https://www.youtube.com/@GPUMODE
- Reference Kernels: https://github.com/gpu-mode/reference-kernels
- Popcorn CLI: https://github.com/gpu-mode/popcorn-cli
- ROCm Docs: https://rocm.docs.amd.com
