# GPU Playground

This repository contains examples and utilities for GPU programming experimenting mainly with Python, C++ and other GPU computing libraries. The examples are inspired by resources from the GPU Mode Group and LeetGPU site.

## Quick Start

### Using Dev Container (Recommended)

The easiest way to get started is using the pre-configured development container:

1. **Prerequisites:**
   - Docker with NVIDIA Container Toolkit installed
   - VS Code with Dev Containers extension
   - NVIDIA GPU with drivers installed on host

2. **Launch:**
   - Open this repository in VS Code
   - Click "Reopen in Container" when prompted
   - The container will automatically set up CUDA 13.0 development environment

For detailed devcontainer setup instructions, see [.devcontainer/README.md](.devcontainer/README.md).

## Prerequisites

To test all the below frameworks ensure the following tools are installed:

- CUDA Toolkit 13.0+ (required for CUDA Tile, CUTLASS CuTe DSL; also for Python, C++, and Mojo development)
  - For CUDA Tile: Requires NVIDIA Driver r580 or later and Blackwell GPU (13.1 release)
  - For CUTLASS: Can be built from source or installed via package manager
- PyTorch or TensorFlow (for Python-based GPU testing)
- A CUDA-capable GPU and appropriate drivers
- CMake and `gcc`/`g++` (for C++ development)
- Mojo SDK (for Mojo development, available from Modular)

## Installing Required Libraries

### For Python:

```bash
pip install torch  # or tensorflow-gpu
pip install tinygrad  # for TinyGrad
pip install triton  # for OpenAI Triton
pip install nvidia-cutlass  # for CUTLASS CuTe DSL
pip install cuda-tile  # for NVIDIA CUDA Tile
```

### For C++ (ensure CUDA is set up):

```bash
sudo apt-get install nvidia-cuda-toolkit
```

### For Mojo:

* Install the Mojo SDK by following instructions from Modular's official documentation.
* Ensure CUDA support is configured for GPU acceleration.

### For CUTLASS (C++):

CUTLASS is a header-only template library for high-performance GEMM operations:

```bash
# Clone CUTLASS repository
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Build with CMake (example for Ampere/Hopper architectures)
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS='80;90a'  # 80=Ampere, 90a=Hopper with arch-accelerated features
make cutlass_profiler -j16

# For selective kernel compilation (faster builds):
cmake .. -DCUTLASS_NVCC_ARCHS='90a' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*gemm_f16_*_nt_align8
```

**Key Points:**
- Header-only library - include `cutlass/include/` in your project's include paths
- Requires C++17 compiler, CMake 3.18+, and CUDA Toolkit 11.4+
- For Hopper (SM90a) and Blackwell (SM100a), use architecture-accelerated targets (note the "a" suffix)
- CUTLASS 4.x introduces **CuTe DSL** - Python interface for writing CUDA kernels without C++ complexity

## Building the Projects

This project uses CMake to manage the build process for all CUDA kernels.

### Prerequisites for C++:
- CUDA Toolkit installed (see [Dev Container](#using-dev-container-recommended) for a pre-configured environment)
- CMake 3.20+
- A C++17 compatible compiler (e.g., GCC 9+)
- Ninja build system (for CMake presets; `sudo apt-get install ninja-build`)

### Build using presets (recommended):

```bash
# Default: Tesla T4 / Compute Capability 7.0
cmake --preset default
cmake --build --preset default -j$(nproc)

# Auto-detect local GPU (requires CMake 3.23+)
cmake --preset native
cmake --build --preset native -j$(nproc)

# Ampere (RTX 3080 / CC 8.6)
cmake --preset ampere
cmake --build --preset ampere -j$(nproc)

# Optimized release build
cmake --preset release
cmake --build --preset release -j$(nproc)
```

Available presets are defined in `CMakePresets.json`.

### Build manually:

```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

By default, this builds for **Tesla T4 (Compute Capability 7.0)**. To build for a different architecture (e.g., RTX 3080/Ampere = 86), use:

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
cmake --build . -j$(nproc)
```

### Build a specific kernel:

```bash
cmake --build . --target gemm
```

### Disable a kernel at configure time:

```bash
cmake -DGPU_ENABLE_GEMM=OFF ..
```

Each kernel has a corresponding `GPU_ENABLE_<KERNEL>` option flag (all `ON` by default).

### Binary locations:

- With presets: `build/<preset>/source/<kernel>/<kernel>` (e.g., `build/default/source/gemm/gemm`)
- With manual build: `build/source/<kernel>/<kernel>` (e.g., `build/source/gemm/gemm`)

### Build on Google Colab

You can also build and run the C++/CUDA kernels on Google Colab using a free GPU runtime. See the [Building on Google Colab](docs/building-on-google-colab.md) guide for full instructions.

## Project Structure

```
gpu_playground/
├── source/                    ← all C++/CUDA source code
│   ├── utils/                 ← shared utility library
│   │   └── cuda_helpers.h     ← CUDA_CHECK macro + getTime()
│   ├── gemm/                  ← each kernel: main.cpp + <kernel>.h + <kernel>.cpp
│   ├── sigmoid/
│   ├── softmax/
│   ├── prefix_sum/
│   ├── geglu/
│   ├── silu/
│   ├── swiglu/
│   ├── matrix_mul/
│   ├── matrix_transpose/
│   ├── spmv/
│   ├── interleave_arrays/
│   ├── reverse_array/
│   ├── value_clipping/
│   └── rgb_to_grayscale/
├── CMakeLists.txt             ← root build file with GPU_ENABLE_* feature flags
├── CMakePresets.json          ← build presets (default, native, ampere, release)
├── cuda_perf_analysis.sh      ← performance profiling script
├── docs/                      ← development guidelines and best practices
├── gpu-mode/                  ← GPU MODE competition tools and submissions
└── .devcontainer/             ← Docker-based CUDA development environment
```

### Kernel Implementations
Each kernel under `source/` follows the same 3-file layout: `main.cpp` (test harness) + `<kernel>.h` (declarations) + `<kernel>.cpp` (implementations).

| Kernel | Description |
|--------|-------------|
| `gemm` | General Matrix Multiplication with advanced tiling (α·A·B + β·C) |
| `sigmoid` | Sigmoid activation — scalar and vectorized float4 variants |
| `softmax` | Numerically stable softmax using multi-stage reduction |
| `prefix_sum` | Parallel inclusive prefix scan (Hillis-Steele) |
| `geglu` | Gated Linear Unit with GELU activation (Transformer FFN) |
| `silu` | SiLU / Swish activation (x·σ(x)) |
| `swiglu` | SwiGLU gated activation |
| `matrix_mul` | Basic tiled matrix multiplication |
| `matrix_transpose` | Matrix transpose using shared memory |
| `spmv` | Sparse Matrix-Vector Multiplication (4 kernel variants) |
| `interleave_arrays` | Array interleaving (AoS ↔ SoA patterns) |
| `reverse_array` | In-place array reversal |
| `value_clipping` | Element-wise value clamping |
| `rgb_to_grayscale` | RGB to grayscale conversion (Rec.601 luminance) |

### GPU MODE Competition
- `gpu-mode/` - GPU MODE kernel competition tools and submissions
  - `SETUP_GUIDE.md` - Complete guide for participating in GPU MODE competitions
  - `tools/` - Helper scripts for setup, submission, and quick reference
  - `submissions/` - Example kernel submissions (e.g., NVFP4 GEMM baseline)

### Documentation
- `docs/` - Development guidelines and best practices
  - `EXAMPLES.md` - Detailed code examples and benchmarks
  - `building-on-google-colab.md` - Step-by-step guide to build and run C++/CUDA kernels on Google Colab
  - `cuda-copilot-rules.md` - CUDA programming rules for AI coding assistants
  - `cuda_best_practices_guide.md` - CUDA best practices and common pitfalls

### Development Tools
- `.devcontainer/` - Docker-based CUDA development environment (VS Code Dev Containers)
- `cuda_perf_analysis.sh` - General-purpose CUDA performance analysis script (auto-detects nvprof/ncu/nsys)

## Examples

For detailed code examples and benchmarks using different frameworks, see [EXAMPLES.md](https://github.com/olibartfast/gpu-playground/blob/master/docs/EXAMPLES.md).

## Further Resources and References

### GPU Programming Communities
* [GPU Mode GitHub](https://github.com/gpu-mode)
* [LeetGPU](https://leetgpu.com)

### Frameworks and Libraries
* [TinyGrad GitHub](https://github.com/geohot/tinygrad)
* [Triton GitHub](https://github.com/openai/triton)
* [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)

### NVIDIA CCCL (CUDA C++ Core Libraries)
* [CCCL GitHub](https://github.com/NVIDIA/cccl) - Unified repository containing:
  - **Thrust** - High-level C++ parallel algorithms library
  - **CUB** - Low-level CUDA building blocks and primitives
  - **libcu++** - CUDA C++ Standard Library (heterogeneous implementation)

### CUTLASS and CuTe
* [CUTLASS GitHub](https://github.com/NVIDIA/cutlass) - High-performance GEMM templates for CUDA
* [CUTLASS Documentation](https://docs.nvidia.com/cutlass)
* [CUTLASS CuTe DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html) - Python interface for writing CUDA kernels
* [CuTe Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
* [NVIDIA CUDA Tile](https://developer.nvidia.com/cuda/tile) - cuTile Python programming language for GPUs

### Other Resources
* [reference-kernels](https://github.com/gpu-mode/reference-kernels) - This repo holds reference kernels for the KernelBot which hosts regular competitions on [discord.gg/gpumode](https://discord.gg/gpumode)
* [Mojo Documentation](https://docs.modular.com/mojo)
* [Modular CUDA Setup Guide](https://www.modular.com/mojo)
* [AI CUDA Engineer: Official Paper and Leaderboard](https://pub.sakana.ai/ai-cuda-engineer)
* [AI CUDA Engineer: Dataset](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)
* [Course on CUDA programming at Oxford Mathematical Institute](https://people.maths.ox.ac.uk/~gilesm/cuda/)

