# GPU playground 

This repository contains examples and utilities for GPU programming experimenting mainly with Python, C++ and other gpu computing libraries. The examples are inspired by resources from the GPU Mode Group and LeetGPU site.

## Prerequisites

To test all the below frameworks ensure the following tools are installed:

- CUDA Toolkit 13.1+ (required for CUDA Tile, CUTLASS CuTe DSL; also for Python, C++, and Mojo development)
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

## Project Structure
- `reverse_array/` - Array reversal examples
- `prefix_sum/` - Parallel prefix sum (scan) implementations
- `matrix_transpose/` - Matrix transpose with shared memory optimization
- `softmax/` - Softmax activation function
- `silu/` - SiLU (Swish) activation function
- `swiglu/` - SwiGLU activation function
- `gemm/` - General Matrix Multiplication (GEMM) with tiled optimization
- other examples incoming, **currently only pure CUDA implementations**...

## Examples

For detailed code examples and benchmarks using different frameworks, see [EXAMPLES.md](EXAMPLES.md).

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
* [Mojo Documentation](https://docs.modular.com/mojo)
* [Modular CUDA Setup Guide](https://www.modular.com/mojo)
* [AI CUDA Engineer: Official Paper and Leaderboard](https://pub.sakana.ai/ai-cuda-engineer)
* [AI CUDA Engineer: Dataset](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)





