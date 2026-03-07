# GPU Playground

A collection of GPU kernel implementations in C++/CUDA and OpenCL, inspired by resources from the [GPU Mode Group](https://github.com/gpu-mode) and [LeetGPU](https://leetgpu.com).

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

- CUDA Toolkit (11.4+ for CUDA builds)
- CMake 3.20+
- A C++17 compatible compiler (e.g., GCC 9+)
- A CUDA-capable GPU and appropriate drivers
- Ninja build system (for CMake presets; `sudo apt-get install ninja-build`)
- For OpenCL C API: `libopencl-dev` / `ocl-icd-opencl-dev`
- For OpenCL C++ wrapper: additionally `opencl-clhpp-headers` (`sudo apt install opencl-clhpp-headers`)

## Installing Required Libraries

### For C++ (ensure CUDA is set up):

```bash
sudo apt-get install nvidia-cuda-toolkit
```

### For Python, Mojo, CUTLASS, and other frameworks:

See [docs/external-frameworks.md](docs/external-frameworks.md) for setup instructions.

## Building the Projects

This project uses CMake to manage the build process for all GPU kernels. Three backends are supported: CUDA (default), OpenCL C API (`USE_OPENCL`), and OpenCL C++ wrapper (`USE_OPENCL_CPP`).

### Build using presets (CUDA, recommended):

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

### Build with OpenCL C API backend:

```bash
cmake -B build/opencl -DUSE_OPENCL=ON
cmake --build build/opencl -j$(nproc)
```

### Build with OpenCL C++ wrapper backend:

```bash
# Requires: sudo apt install opencl-clhpp-headers
cmake -B build/opencl_cpp -DUSE_OPENCL_CPP=ON
cmake --build build/opencl_cpp -j$(nproc)
```

Binaries land in `build/opencl/source/<kernel>/<kernel>` or `build/opencl_cpp/source/<kernel>/<kernel>`. The same `GPU_ENABLE_*` feature flags apply.

### Build manually (CUDA):

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
- OpenCL C API build: `build/opencl/source/<kernel>/<kernel>`
- OpenCL C++ build: `build/opencl_cpp/source/<kernel>/<kernel>`

### Build on Google Colab

You can also build and run the C++/CUDA kernels on Google Colab using a free GPU runtime. See the [Building on Google Colab](docs/building-on-google-colab.md) guide for full instructions.

## Project Structure

```
gpu_playground/
├── source/                    ← all C++/CUDA/OpenCL source code
│   ├── utils/                 ← shared utility libraries
│   │   ├── cuda_helpers.h         ← CUDA_CHECK macro + getTime()
│   │   ├── opencl_c_helpers.h     ← C API: CL_CHECK + clSetupGPU/clBuildFromSource/clTeardown
│   │   └── opencl_helpers.h       ← C++ wrapper: clppGetGPUDevice/clppBuildProgram/clppPreferredLocalSize
│   ├── gemm/                  ← each kernel: main.cpp + cuda/ + opencl/ + opencl_cpp/
│   │   ├── main.cpp           ← backend-agnostic test harness (3-way #ifdef)
│   │   ├── cuda/              ← CUDA kernel + host-pointer wrapper
│   │   ├── opencl/            ← OpenCL C API implementation
│   │   └── opencl_cpp/        ← OpenCL C++ wrapper implementation
│   └── ...                    ← same layout for all 14 kernels
├── CMakeLists.txt             ← root build file; USE_OPENCL / USE_OPENCL_CPP + GPU_ENABLE_* flags
├── CMakePresets.json          ← build presets (default, native, ampere, release)
├── cuda_perf_analysis.sh      ← performance profiling script
├── docs/                      ← development guidelines and best practices
├── gpu-mode/                  ← GPU MODE competition tools and submissions
└── .devcontainer/             ← Docker-based CUDA development environment
```

### Kernel Implementations

Each kernel under `source/` has `cuda/`, `opencl/`, and `opencl_cpp/` subdirectories for the backend implementations, and a shared `main.cpp` test harness that selects the backend at compile time.

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
  - `adding-a-new-kernel.md` - How-To adding a new kernel to the project
  - `external-frameworks.md` - Setup instructions for external frameworks (CUTLASS, Mojo, Triton, TinyGrad)

### Development Tools
- `.devcontainer/` - Docker-based CUDA development environment (VS Code Dev Containers)
- `cuda_perf_analysis.sh` - General-purpose CUDA performance analysis script (auto-detects nvprof/ncu/nsys)

## Examples

For detailed code examples and benchmarks, see [EXAMPLES.md](https://github.com/olibartfast/gpu-playground/blob/master/docs/EXAMPLES.md).

## Further Resources and References

### GPU Programming Communities
* [GPU Mode GitHub](https://github.com/gpu-mode)
* [LeetGPU](https://leetgpu.com)
* [reference-kernels](https://github.com/gpu-mode/reference-kernels) - Reference kernels for the KernelBot competitions on [discord.gg/gpumode](https://discord.gg/gpumode)
* [Course on CUDA programming at Oxford Mathematical Institute](https://people.maths.ox.ac.uk/~gilesm/cuda/)

### Frameworks and Libraries
* [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
* [TinyGrad GitHub](https://github.com/geohot/tinygrad)
* [Triton GitHub](https://github.com/openai/triton)
* [CUTLASS GitHub](https://github.com/NVIDIA/cutlass) - High-performance GEMM templates for CUDA
* [NVIDIA CUDA Tile](https://developer.nvidia.com/cuda/tile) - cuTile Python programming language for GPUs
* [Mojo Documentation](https://docs.modular.com/mojo)

For setup instructions for these frameworks, see [docs/external-frameworks.md](docs/external-frameworks.md).

### NVIDIA CCCL (CUDA C++ Core Libraries)
* [CCCL GitHub](https://github.com/NVIDIA/cccl) - Unified repository containing:
  - **Thrust** - High-level C++ parallel algorithms library
  - **CUB** - Low-level CUDA building blocks and primitives
  - **libcu++** - CUDA C++ Standard Library (heterogeneous implementation)

### AI-Assisted CUDA Development
* [AI CUDA Engineer: Official Paper and Leaderboard](https://pub.sakana.ai/ai-cuda-engineer)
* [AI CUDA Engineer: Dataset](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)
* [ByteDance CUDA Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent)
