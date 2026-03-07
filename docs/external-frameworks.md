# External Frameworks Setup

This page covers setup instructions for GPU frameworks referenced in this repository's resources but not yet implemented as kernels here. They are useful for experimentation and comparison.

## Python Frameworks

```bash
pip install torch              # PyTorch (or: pip install tensorflow for TensorFlow)
pip install tinygrad           # TinyGrad
pip install triton             # OpenAI Triton
pip install nvidia-cutlass     # CUTLASS CuTe DSL (Python)
pip install cuda-tile          # NVIDIA CUDA Tile
```

**Prerequisites for specific packages:**
- `nvidia-cutlass` / `cuda-tile`: CUDA Toolkit 13.0+
- `cuda-tile`: NVIDIA Driver r580 or later and a Blackwell GPU (13.1 release)

## Mojo

* Install the Mojo SDK by following instructions from [Modular's official documentation](https://docs.modular.com/mojo).
* Ensure CUDA support is configured for GPU acceleration.

## CUTLASS (C++)

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
- Header-only library — include `cutlass/include/` in your project's include paths
- Requires C++17 compiler, CMake 3.18+, and CUDA Toolkit 11.4+
- For Hopper (SM90a) and Blackwell (SM100a), use architecture-accelerated targets (note the `a` suffix)
- CUTLASS 4.x introduces **CuTe DSL** — Python interface for writing CUDA kernels without C++ complexity

## Further Reading

* [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
* [CUTLASS Documentation](https://docs.nvidia.com/cutlass)
* [CUTLASS CuTe DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html)
* [CuTe Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
* [NVIDIA CUDA Tile](https://developer.nvidia.com/cuda/tile)
* [TinyGrad GitHub](https://github.com/geohot/tinygrad)
* [Triton GitHub](https://github.com/openai/triton)
* [Mojo Documentation](https://docs.modular.com/mojo)
* [Modular CUDA Setup Guide](https://www.modular.com/mojo)
