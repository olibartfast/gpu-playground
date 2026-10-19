# External Frameworks Setup

> Audience: humans · Applies to: all · Last reviewed: 2026-10

This page covers frameworks that this repository references for comparison and
experimentation, but does not implement as kernels. The repository's own kernels
do not depend on them.

## Python Frameworks

```bash
pip install torch              # PyTorch (or: pip install tensorflow for TensorFlow)
pip install tinygrad           # TinyGrad
pip install triton             # OpenAI Triton
pip install nvidia-cutlass     # CUTLASS CuTe DSL (Python)
pip install cuda-tile          # NVIDIA CUDA Tile
```

Package requirements:

- `nvidia-cutlass` and `cuda-tile` require CUDA Toolkit 13.0 or newer.
- `cuda-tile` additionally requires NVIDIA driver r580 or newer and a Blackwell
  GPU (13.1 release).

## Mojo

- Install the Mojo SDK following [Modular's official documentation](https://docs.modular.com/mojo).
- Configure CUDA support for GPU acceleration.

## CUTLASS (C++)

CUTLASS is a header-only template library for high-performance GEMM:

```bash
# Clone the CUTLASS repository
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Build with CMake (example for Ampere and Hopper)
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS='80;90a'  # 80=Ampere, 90a=Hopper with architecture-accelerated features
make cutlass_profiler -j16

# For selective kernel compilation (faster builds):
cmake .. -DCUTLASS_NVCC_ARCHS='90a' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*gemm_f16_*_nt_align8
```

Key points:

- It is header-only: add `cutlass/include/` to your include path.
- It requires a C++17 compiler, CMake 3.18 or newer, and CUDA Toolkit 11.4 or
  newer.
- Hopper (SM90a) and Blackwell (SM100a) use architecture-accelerated targets;
  note the `a` suffix.
- CUTLASS 4.x introduces the **CuTe DSL**, a Python interface for writing CUDA
  kernels without C++ template code.

## Further Reading

* [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
* [CUTLASS Documentation](https://docs.nvidia.com/cutlass)
* [CUTLASS CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)
* [CuTe Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
* [NVIDIA CUDA Tile](https://developer.nvidia.com/cuda/tile)
* [TinyGrad GitHub](https://github.com/geohot/tinygrad)
* [Triton GitHub](https://github.com/openai/triton)
* [Mojo Documentation](https://docs.modular.com/mojo)
* [Modular CUDA Setup Guide](https://www.modular.com/mojo)
