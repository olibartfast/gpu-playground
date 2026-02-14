# Building gpu-playground on Google Colab

## Prerequisites

Before building, make sure you have a **GPU runtime** enabled:

1. Go to **Runtime → Change runtime type**
2. Select **T4 GPU** (free tier) or a higher-tier GPU if available
3. Click **Save**

Verify the GPU is available:

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

## Setting the Right CUDA Architecture

The `-DCMAKE_CUDA_ARCHITECTURES` flag must match your Colab GPU. Common values:

| GPU | Compute Capability | Flag |
|-----|-------------------|------|
| Tesla T4 (free tier) | 7.5 | `-DCMAKE_CUDA_ARCHITECTURES=75` |
| A100 | 8.0 | `-DCMAKE_CUDA_ARCHITECTURES=80` |
| L4 | 8.9 | `-DCMAKE_CUDA_ARCHITECTURES=89` |

You can detect it programmatically:

```bash
!python3 -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True, text=True)
cc = result.stdout.strip().replace('.', '')
print(f'Detected compute capability: {result.stdout.strip()} → use -DCMAKE_CUDA_ARCHITECTURES={cc}')
"
```

## Building a Single Target

To build and run only a specific kernel (e.g., `softmax`):

```bash
%cd /content/gpu-playground/build
!cmake --build . --target softmax
!./softmax/softmax
```

Available targets include: `reverse_array`, `prefix_sum`, `matrix_transpose`, `matrix_mul`, `gemm`, `spmv`, `softmax`, `geglu`, `silu`, `swiglu`, `interleave_arrays`, `value_clipping`.

## Persisting Builds Across Sessions

Colab resets the filesystem on session restart. To avoid rebuilding every time, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone to Drive (first time only)
!git clone https://github.com/olibartfast/gpu-playground.git /content/drive/MyDrive/gpu-playground

# Build from Drive
%cd /content/drive/MyDrive/gpu-playground
!mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build . -j$(nproc)
```

> **Note:** Building on Google Drive is slower due to network filesystem overhead. An alternative approach is to keep the source on Drive and copy it to local `/content` for building.

## Profiling (Optional)

The repo includes `cuda_perf_analysis.sh` for profiling with Nsight Compute. Check availability first:

```bash
!which ncu && ncu --version || echo "Nsight Compute not available in this Colab runtime"
```

If `ncu` is available:

```bash
%cd /content/gpu-playground
!bash cuda_perf_analysis.sh ./build/gemm/gemm
```

As an alternative, `nvprof` (legacy) or basic CUDA event timing within the kernels can be used for benchmarking.

## Troubleshooting

**CMake not found or too old:**

```bash
!apt-get update && apt-get install -y cmake
!cmake --version  # should be 3.20+
```

**CUDA toolkit not found:**

```bash
!nvcc --version
# If missing (unlikely on GPU runtime):
!apt-get install -y nvidia-cuda-toolkit
```

**Wrong GPU architecture (illegal instruction at runtime):**

Double-check the compute capability with `nvidia-smi` and rebuild with the correct `-DCMAKE_CUDA_ARCHITECTURES` value.

**Session disconnected mid-build:**

Colab sessions can disconnect after ~90 minutes of inactivity (free tier). For long builds, keep the browser tab active or use the Google Drive persistence approach described above.
