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

### Build all projects:

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

See the root `CMakeLists.txt` for additional architecture configuration options and best practices.

### Build a specific project:

To build only one specific kernel (e.g., `gemm`), you can specify the target:

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --target gemm
```

The compiled binaries will be located in their respective subdirectories under `build/` (e.g., `build/gemm/gemm`).

### Build on Google Colab

You can also build and run the C++/CUDA kernels on Google Colab using a free GPU runtime. See the [Building on Google Colab](docs/building-on-google-colab.md) guide for full instructions.

## Project Structure

### Kernel Implementations
Each directory below contains a standalone CUDA implementation and its own `CMakeLists.txt`.

- `reverse_array/` - Simple array reversal
- `prefix_sum/` - Parallel prefix sum (scan)
- `matrix_transpose/` - Optimized matrix transpose using shared memory
- `matrix_mul/` - Basic tiled matrix multiplication
- `gemm/` - General Matrix Multiplication with advanced tiling
- `spmv/` - Sparse Matrix-Vector Multiplication
- `softmax/` - Softmax activation function
- `geglu/` - Gated Linear Unit with GELU activation
- `silu/` - SiLU (Swish) activation function
- `swiglu/` - SwiGLU activation function
- `interleave_arrays/` - Array interleaving (AoS to SoA patterns)
- `value_clipping/` - Element-wise value clipping

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
- `.devcontainer/` - Docker-based CUDA development environment (VS Code)
- `cuda_perf_analysis.sh` - General-purpose CUDA performance analysis script
- `tests/` - Test infrastructure (WIP)
- `utils/` - Utility functions (WIP)

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

