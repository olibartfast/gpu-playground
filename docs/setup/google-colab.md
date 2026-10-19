# Building on Google Colab

> Audience: humans · Applies to: CUDA · Last reviewed: 2026-10

Google Colab's free GPU runtime can build and run this repository's CUDA
kernels. Select a GPU runtime, clone the repository, configure CMake for that
GPU's compute capability, build, and run a binary.

The notebook workflow below needs no local setup. If you prefer a terminal, the
[Colab CLI](https://github.com/googlecolab/google-colab-cli) provisions and
controls the same runtimes over SSH.

## Install the Colab CLI (Optional)

The Colab CLI supports Linux and macOS; Windows is not supported. Install it
with `uv` (recommended upstream) or `pip`:

```bash
uv tool install google-colab-cli
# or
pip install google-colab-cli
```

Confirm the install, provision a GPU runtime, and open a remote shell:

```bash
colab version
colab new -s gpu-playground --gpu T4
colab ssh -s gpu-playground
```

Inside the remote shell, clone and build without the notebook `!`/`%` prefixes:

```bash
git clone https://github.com/olibartfast/gpu-playground.git
cmake -S gpu-playground -B gpu-playground/build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build gpu-playground/build -j"$(nproc)"
./gpu-playground/build/source/softmax/softmax
```

Exit the remote shell and release the runtime when finished:

```bash
exit
colab stop -s gpu-playground
```

`colab sessions` lists active runtimes, and `colab run --gpu T4 script.py`
provisions a VM for a one-off Python script and tears it down automatically.
Runtime availability and quotas follow your Colab account.

## Choose a GPU Runtime

1. Open **Runtime → Change runtime type**.
2. Select **T4 GPU** (free tier) or a higher-tier GPU if available.
3. Click **Save**.

Confirm that the GPU is visible:

```bash
!nvidia-smi
```

## Clone and Build

```bash
# Clone the repository
!git clone https://github.com/olibartfast/gpu-playground.git

# Create build directory and configure
!mkdir -p gpu-playground/build
%cd gpu-playground/build

# Configure with CMake (T4 = compute capability 7.5)
!cmake .. -DCMAKE_CUDA_ARCHITECTURES=75

# Build all targets
!cmake --build . -j$(nproc)
```

`CMAKE_CUDA_ARCHITECTURES` must match the session's GPU. Otherwise the build
succeeds but the binary fails at runtime with an invalid instruction.

## Set the CUDA Architecture

Common Colab GPUs and their flags:

| GPU | Compute capability | Flag |
|-----|--------------------|------|
| Tesla T4 (free tier) | 7.5 | `-DCMAKE_CUDA_ARCHITECTURES=75` |
| A100 | 8.0 | `-DCMAKE_CUDA_ARCHITECTURES=80` |
| L4 | 8.9 | `-DCMAKE_CUDA_ARCHITECTURES=89` |

You can also detect the value programmatically:

```bash
!python3 -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True, text=True)
cc = result.stdout.strip().replace('.', '')
print(f'Detected compute capability: {result.stdout.strip()} → use -DCMAKE_CUDA_ARCHITECTURES={cc}')
"
```

## Build and Run a Single Kernel

Building one target is faster. For example, to build and run `softmax`:

```bash
%cd /content/gpu-playground/build
!cmake --build . --target softmax
!./source/softmax/softmax
```

Every binary lands under `build/source/<kernel>/<kernel>`:

```bash
# Examples
!./source/gemm/gemm
!./source/sigmoid/sigmoid
!./source/softmax/softmax
!./source/spmv/spmv
```

Common targets:

| Target | Binary path (relative to `build/`) |
|--------|------------------------------------|
| `reverse_array` | `source/reverse_array/reverse_array` |
| `prefix_sum` | `source/prefix_sum/prefix_sum` |
| `matrix_transpose` | `source/matrix_transpose/matrix_transpose` |
| `matrix_mul` | `source/matrix_mul/matrix_mul` |
| `gemm` | `source/gemm/gemm` |
| `sigmoid` | `source/sigmoid/sigmoid` |
| `softmax` | `source/softmax/softmax` |
| `spmv` | `source/spmv/spmv` |
| `geglu` | `source/geglu/geglu` |
| `silu` | `source/silu/silu` |
| `swiglu` | `source/swiglu/swiglu` |
| `interleave_arrays` | `source/interleave_arrays/interleave_arrays` |
| `value_clipping` | `source/value_clipping/value_clipping` |
| `rgb_to_grayscale` | `source/rgb_to_grayscale/rgb_to_grayscale` |

The [kernel inventory](../../Readme.md#kernel-inventory) lists every target,
including the CUDA-only kernels added for LeetGPU challenges.

## Disable Kernels to Shorten the Build

Each kernel has a `GPU_ENABLE_<KERNEL>` flag, `ON` by default. Disable the ones
you do not need to cut build time:

```bash
!cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 -DGPU_ENABLE_SPMV=OFF -DGPU_ENABLE_GEMM=OFF
!cmake --build . -j$(nproc)
```

## Persist Builds Across Sessions

Colab resets its filesystem when a session ends. Keep the checkout on Google
Drive to avoid rebuilding:

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone to Drive (first time only)
!git clone https://github.com/olibartfast/gpu-playground.git /content/drive/MyDrive/gpu-playground

# Build from Drive
%cd /content/drive/MyDrive/gpu-playground
!mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build . -j$(nproc)
```

> **Note:** Building on a Drive mount is slower because every file operation
> crosses a network filesystem. Keep the source on Drive and copy it to local
> `/content` for builds.

## Profile a Kernel (Optional)

`cuda_perf_analysis.sh` auto-detects an available profiler. Check for Nsight
Compute:

```bash
!which ncu && ncu --version || echo "Nsight Compute not available in this Colab runtime"
```

If `ncu` is available:

```bash
%cd /content/gpu-playground
!bash cuda_perf_analysis.sh ./build/source/gemm/gemm
```

Otherwise, the harness's CUDA event timing still measures kernels; legacy
`nvprof` may also be available.

## Troubleshooting

**CMake is missing or too old:**

```bash
!apt-get update && apt-get install -y cmake
!cmake --version  # should be 3.20+
```

**The CUDA toolkit is missing:**

```bash
!nvcc --version
# If missing (unlikely on a GPU runtime):
!apt-get install -y nvidia-cuda-toolkit
```

**The binary fails at runtime with an illegal instruction:**

The architecture flag does not match the assigned GPU. Check the compute
capability with `nvidia-smi` and rebuild with the correct
`-DCMAKE_CUDA_ARCHITECTURES` value.

**The session disconnects mid-build:**

Free-tier sessions can disconnect after roughly 90 minutes of inactivity. Keep
the tab active during long builds, or use the Drive persistence approach above.
