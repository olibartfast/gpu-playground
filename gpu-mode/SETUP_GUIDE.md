# GPU Mode kernels - Local PC Setup Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Install Rust and Cargo](#step-1-install-rust-and-cargo)
4. [Step 2: Install Popcorn CLI](#step-2-install-popcorn-cli)
5. [Step 3: Join GPU MODE Discord](#step-3-join-gpu-mode-discord)
6. [Step 4: Authenticate Popcorn CLI](#step-4-authenticate-popcorn-cli)
7. [Step 5: Clone Reference Kernels](#step-5-clone-reference-kernels)
8. [Step 6: Understand the Problem](#step-6-understand-the-problem)
9. [Step 7: Create Your First Submission](#step-7-create-your-first-submission)
10. [Step 8: Submit to Leaderboard](#step-8-submit-to-leaderboard)
11. [Step 9: Check Your Position](#step-9-check-your-position)
12. [Workflow Summary](#workflow-summary)
13. [Optimization Strategies](#optimization-strategies)
14. [Learning Resources](#learning-resources)
15. [Troubleshooting](#troubleshooting)
16. [Quick Reference Commands](#quick-reference-commands)

---

## Overview

This guide shows how to set up your local PC for GPU Mode kernel submissions.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or WSL2 on Windows
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (if running locally with GPU)
- **Rust**: Latest stable version (for building Popcorn CLI)
- **Git**: For cloning repositories

### Accounts Required
1. **Discord Account** - Join GPU MODE Discord: https://discord.gg/gpumode
2. **GitHub Account** (optional) - Alternative authentication method

---

## Step 1: Install Rust and Cargo

Rust is required to build the Popcorn CLI submission tool.

### Linux/macOS:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the prompts, then source the environment
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### Windows (WSL2):
Same as Linux instructions above.

---

## Step 2: Install Popcorn CLI

Popcorn CLI is the submission tool.

```bash
# Create a workspace directory (call it the name you prefer, here below referenced as WORKSPACE)
export WORKSPACE=gpu-mode
mkdir -p ~/$WORKSPACE
cd ~/$WORKSPACE

# Clone the Popcorn CLI repository
git clone https://github.com/gpu-mode/popcorn-cli.git
cd popcorn-cli

# Build from source
./build.sh

# The binary will be at: target/release/popcorn-cli
```

### Add to PATH (recommended):
```bash
# Option 1: Copy to system bin
sudo cp target/release/popcorn-cli /usr/local/bin/

# Option 2: Add alias to your shell config
echo 'alias popcorn="~/$WORKSPACE/popcorn-cli/target/release/popcorn-cli"' >> ~/.bashrc
source ~/.bashrc
```

### Verify Installation:
```bash
popcorn-cli --help
# or if not in PATH:
~/${WORKSPACE}/popcorn-cli/target/release/popcorn-cli --help
```

---

## Step 3: Join GPU MODE Discord

1. Go to https://discord.gg/gpumode
2. Accept the invite and join the server
3. Browse available channels for your area of interest:
   - `#general` - General discussions
   - `#nvidia-competition` - NVIDIA hackathon/competition channel (if active)
   - Problem-specific channels for different kernels
4. Read the pinned messages for announcements and updates

---

## Step 4: Authenticate Popcorn CLI

### Get API URL from Discord:
1. In the GPU MODE Discord server, in any text channel where you can type (try `#general` or a problem-specific channel), type:
   ```
   /get-api-url
   ```
2. The bot will respond with your API URL (looks like):
   ```
   API URL: https://discord-cluster-manager-<some-id>.herokuapp.com
   ```

### Set Environment Variable:
```bash
# Add to your shell (replace <url> with the actual URL from Discord)
export POPCORN_API_URL="<url_from_discord>"

# Make it persistent (add to ~/.bashrc or ~/.zshrc)
echo 'export POPCORN_API_URL="<url_from_discord>"' >> ~/.bashrc
source ~/.bashrc
```

### Register with Discord (Recommended):
```bash
cd ~/$WORKSPACE/popcorn-cli
./target/release/popcorn-cli register discord
```

This will:
- Open a browser window for Discord OAuth
- Ask you to authorize the application
- Save your CLI ID to `~/.popcorn.yaml`

### Verify Authentication:
```bash
cat ~/.popcorn.yaml
# Should show your cli_id
```

### Alternative: Register with GitHub:
```bash
./target/release/popcorn-cli register github
```

---

## Step 5: Clone Reference Kernels

The reference kernels repository contains problem specifications, baseline implementations, and test cases.

```bash
cd ~/$WORKSPACE
git clone https://github.com/gpu-mode/reference-kernels.git
cd reference-kernels

# Navigate to NVIDIA hackathon problems
cd problems/nvidia
ls
# You should see: nvfp4_gemm, nvfp4_gemv, eval.py, utils.py
```

### Explore a Problem:
```bash
cd nvfp4_gemm
ls
# Files you'll see:
# - reference.py      : PyTorch reference implementation
# - template.py       : Baseline kernel to start from
# - submission.py     : Example submission format
# - task.yml          : Problem specification (shapes, constraints)
# - task.py           : Task runner
# - utils.py          : Helper functions
```

---

## Step 6: Understand the Problem

> **Note:** Different problems are available on the GPU MODE platform. The example below shows NVFP4 GEMM, but you should refer to the specific problem you're working on in the reference-kernels repository.

### Example: NVFP4 GEMM

**What to implement:**
```python
C = A @ B.T  # Matrix multiplication with NVFP4 block scaling
```

**Inputs:**
- `a`: M × K × L in NVFP4 (float4_e2m1fn) — 4-bit floats
- `b`: N × K × L in NVFP4 (float4_e2m1fn)
- `sfa`: M × (K/16) × L scale factors in FP8 — every 16 elements share one scale
- `sfb`: N × (K/16) × L scale factors in FP8
- `sfa_permuted`, `sfb_permuted`: Pre-permuted scale factors for easier access
- `c`: M × N × L output buffer (FP16)

**Constraints:**
- M, N divisible by MMA tile size
- K divisible by 256
- Target hardware: NVIDIA B200 (Blackwell)

**Key concepts:**
- **NVFP4**: 4-bit floating point format (2-bit exponent, 1-bit mantissa)
- **Block Scaling**: Every 16 FP4 values share one FP8 scale factor
- **Goal**: Optimize for maximum throughput on Blackwell tensor cores

---

## Step 7: Create Your First Submission

### Directory Structure (recommended):
```bash
cd ~/$WORKSPACE
mkdir submissions
cd submissions
```

### Copy Template:
```bash
cp ~/$WORKSPACE/reference-kernels/problems/nvidia/nvfp4_gemm/template.py \
   ~/$WORKSPACE/submissions/my_nvfp4_gemm.py
```

### Or create from scratch:
See the example submissions in the `submissions/` folder of this repo.

---

## Step 8: Submit to Leaderboard

### Browse Available Leaderboards:
```bash
cd ~/$WORKSPACE/popcorn-cli
./target/release/popcorn-cli submit
# This opens an interactive TUI to browse leaderboards
```

### Submit Your Kernel:
```bash
cd ~/$WORKSPACE/popcorn-cli
./target/release/popcorn-cli submit \
  --gpu <GPU MODEL> \
  --leaderboard nvfp4_gemm \
  --mode leaderboard \
  ~/$WORKSPACE/submissions/my_nvfp4_gemm.py
```

**Parameters:**
- `--gpu <GPU_MODEL>`: Target hardware (e.g., B200, A100, H100 - check leaderboard for available options)
- `--leaderboard <name>`: Problem/leaderboard name (e.g., nvfp4_gemm)
- `--mode leaderboard`: Submission mode (use `dev` for testing, `leaderboard` for official submissions)
- Final argument: Path to your submission file

---

## Step 9: Check Your Position

### Discord Command:
In GPU MODE Discord, type:
```
/leaderboard nvfp4_gemm
```

### GPU MODE Website:
Visit https://www.gpumode.com for live leaderboard standings.

---

## Workflow Summary

```bash
# 1. Set up environment once
export POPCORN_API_URL="your_api_url_from_discord"

# 2. Work on your kernel
cd ~/$WORKSPACE/submissions
vim my_kernel.py  # or your preferred editor

# 3. Test locally (optional, requires GPU)
python my_kernel.py

# 4. Submit to leaderboard
cd ~/$WORKSPACE/popcorn-cli
./target/release/popcorn-cli submit \
  --gpu <GPU MODEL> \
  --leaderboard nvfp4_gemm \
  --mode leaderboard \
  ~/$WORKSPACE/submissions/my_kernel.py

# 5. Check results
# - View terminal output for benchmark results
# - Check Discord: /leaderboard nvfp4_gemm
# - Visit https://www.gpumode.com
```

---

## Optimization Strategies

### Level 1: Baseline (torch._scaled_mm)
Use PyTorch's built-in `torch._scaled_mm` which calls cuBLAS/CUTLASS:
- ✅ Easy to implement
- ✅ Correct
- ❌ Not competitive (far from speed of light)

### Level 2: GPU-Optimized PyTorch
- Remove CPU transfers (`.cpu()`, `.numpy()`)
- Use pre-permuted scale factors
- JIT compile with `@torch.jit.script`
- Batch operations efficiently

### Level 3: Triton Kernel
Use OpenAI Triton to write GPU kernels in Python:
- `tl.dot()` for matrix multiplication
- `tl.load()` / `tl.store()` for memory operations
- Manual tiling and blocking

### Level 4: CuTeDSL / CUTLASS
Write kernels targeting Blackwell-specific hardware:
- **TCGen05**: 5th-gen Tensor Cores with native NVFP4 support
- **TMA**: Tensor Memory Accelerator for async data movement
- **TMEM**: Fast on-chip accumulator storage
- **Warp Specialization**: Different warps do different tasks
- **Persistent Kernels**: Stay resident on GPU between tiles

**Resources:**
- CUTLASS repo: https://github.com/NVIDIA/cutlass
- Examples: `cutlass/examples/python/CuTeDSL/blackwell/`
- Key file: `dense_blockscaled_gemm_persistent.py`

---

## Learning Resources

### Official Resources:
- **GPU MODE YouTube**: https://www.youtube.com/@GPUMODE
  - Weekly lectures from NVIDIA engineers and ML researchers
  - CuTeDSL tutorials
  - Kernel optimization techniques
- **NVIDIA NVFP4 Blog**: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- **CUTLASS Documentation**: https://github.com/NVIDIA/cutlass

### Community:
- **GPU MODE Discord**: https://discord.gg/gpumode
  - Problem-specific channels - For questions and discussions about specific kernels
  - `#general` - General GPU programming discussions
  - Competition channels - For active competitions/hackathons
  - Ask questions, share tips, learn from others

### Example Kernels:
- Check the reference-kernels repo for baseline implementations
- Study top submissions on the leaderboard
- GPU MODE Discord has shared examples and tips

---

## Troubleshooting

### Authentication Issues (401 Unauthorized):
```bash
# Remove old config
rm ~/.popcorn.yaml

# Re-register
cd ~/$WORKSPACE/popcorn-cli
./target/release/popcorn-cli register discord (or ./target/release/popcorn-cli reregister discord)

# Verify
cat ~/.popcorn.yaml
```

### Popcorn CLI Build Errors:
```bash
# Update Rust
rustup update stable

# Clean and rebuild
cd ~/$WORKSPACE/popcorn-cli
cargo clean
./build.sh

# Or regenerate Cargo.lock
rm Cargo.lock
./build.sh
```

### Can't Find Leaderboard:
- Use the TUI: `./target/release/popcorn-cli submit` (no arguments)
- Ask in Discord: `/leaderboard` to see all available
- The exact name might differ: `nvfp4-gemm`, `nvidia_nvfp4_gemm`, etc.

### Submission Validation Errors:
- Ensure your kernel follows the expected signature
- Check `task.py` in the problem folder for the exact interface
- Test locally first if you have a GPU
- Ask in `#nvidia-competition` on Discord

---

## Quick Reference Commands

```bash
# Set API URL (get from Discord: /get-api-url)
export POPCORN_API_URL="<url_from_discord>"

# Register
~/$WORKSPACE/popcorn-cli/target/release/popcorn-cli register discord

# Browse leaderboards
~/$WORKSPACE/popcorn-cli/target/release/popcorn-cli submit

# Submit kernel
~/$WORKSPACE/popcorn-cli/target/release/popcorn-cli submit \
  --gpu <MODEL> \
  --leaderboard nvfp4_gemm \
  --mode leaderboard \
  ~/$WORKSPACE/submissions/my_kernel.py

# Check leaderboard (Discord)
/leaderboard nvfp4_gemm
```

---

**Questions?** Ask in the appropriate GPU MODE Discord channel!

**Additional Resources:**
- GPU MODE Website: https://www.gpumode.com
- GPU MODE YouTube: https://www.youtube.com/@GPUMODE
- Reference Kernels: https://github.com/gpu-mode/reference-kernels
- Popcorn CLI: https://github.com/gpu-mode/popcorn-cli
