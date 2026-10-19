# GPU Playground

A collection of standalone GPU kernels written in C++/CUDA and OpenCL, built as a
learning and benchmarking playground. Each kernel pairs a CPU reference with one
or more GPU backends and a shared test harness, so correctness and performance
can be compared directly. The project is inspired by resources from the
[GPU Mode Group](https://github.com/gpu-mode) and [LeetGPU](https://leetgpu.com).

## Quick Start

### Using the dev container (recommended)

The pre-configured development container is the fastest way to get a working
CUDA toolchain:

1. **Prerequisites**
   - Docker with the NVIDIA Container Toolkit installed
   - VS Code with the Dev Containers extension
   - An NVIDIA GPU and driver installed on the host

2. **Launch**
   - Open this repository in VS Code.
   - Click "Reopen in Container" when prompted.
   - The container sets up a CUDA 13.0 development environment automatically.

For details, see [.devcontainer/README.md](.devcontainer/README.md).

## Prerequisites

- CUDA Toolkit 11.4 or newer for CUDA builds
- CMake 3.20 or newer
- A C++17 compiler, such as GCC 9 or newer
- A CUDA-capable GPU with matching drivers
- Ninja, required by the CMake presets (`sudo apt-get install ninja-build`)
- For the OpenCL C API: `libopencl-dev` or `ocl-icd-opencl-dev`
- For the OpenCL C++ wrapper: additionally `opencl-clhpp-headers`
  (`sudo apt install opencl-clhpp-headers`)

To install the CUDA toolkit on Ubuntu:

```bash
sudo apt-get install nvidia-cuda-toolkit
```

For Python, Mojo, CUTLASS, and other frameworks, see
[docs/setup/external-frameworks.md](docs/setup/external-frameworks.md).

## Building

Three backends are supported: CUDA (the default), the OpenCL C API (`USE_OPENCL`),
and the OpenCL C++ wrapper (`USE_OPENCL_CPP`). All are driven by CMake.

### CUDA presets (recommended)

```bash
# Default: Tesla T4 / Compute Capability 7.0
cmake --preset default
cmake --build --preset default -j$(nproc)

# Auto-detect the local GPU (requires CMake 3.23+)
cmake --preset native
cmake --build --preset native -j$(nproc)

# Ampere (RTX 3080 / CC 8.6)
cmake --preset ampere
cmake --build --preset ampere -j$(nproc)

# Optimized release build
cmake --preset release
cmake --build --preset release -j$(nproc)
```

The available presets are defined in `CMakePresets.json`.

### OpenCL backends

OpenCL C API:

```bash
cmake -B build/opencl -DUSE_OPENCL=ON
cmake --build build/opencl -j$(nproc)
```

OpenCL C++ wrapper:

```bash
# Requires: sudo apt install opencl-clhpp-headers
cmake -B build/opencl_cpp -DUSE_OPENCL_CPP=ON
cmake --build build/opencl_cpp -j$(nproc)
```

Both honor the same `GPU_ENABLE_*` feature flags as the CUDA build.

### Manual CMake build

```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

This builds for **Tesla T4 (Compute Capability 7.0)** by default. To target a
different architecture, pass it explicitly — for example, an RTX 3080 on Ampere:

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
cmake --build . -j$(nproc)
```

### Building selected kernels

Every kernel has a `GPU_ENABLE_<KERNEL>` option, `ON` by default. Build one
target directly:

```bash
cmake --build . --target gemm
```

Or exclude kernels at configure time to shorten the build:

```bash
cmake -DGPU_ENABLE_GEMM=OFF ..
```

### Binary locations

- Preset builds: `build/<preset>/source/<kernel>/<kernel>`
  (for example, `build/default/source/gemm/gemm`)
- Manual builds: `build/source/<kernel>/<kernel>`
- OpenCL C API: `build/opencl/source/<kernel>/<kernel>`
- OpenCL C++ wrapper: `build/opencl_cpp/source/<kernel>/<kernel>`

### Building on Google Colab

The C++/CUDA kernels also build on a free Colab GPU runtime, from a notebook or
from a terminal with the Colab CLI. See
[Building on Google Colab](docs/setup/google-colab.md) for the full walkthrough,
including CLI installation.

## Running Kernels

Run a kernel binary directly; each one validates its output against the CPU
reference and returns a non-zero exit code on failure.

The LeetGPU-compatible FP16 dot product, categorical cross-entropy, and Gaussian
blur kernels also provide a `--performance` mode with the challenge's benchmark
shape:

```bash
# FP16 dot product: functional cases and a 100-million-element shape
./build/default/source/fp16_dot_product/fp16_dot_product
./build/default/source/fp16_dot_product/fp16_dot_product --performance

# Categorical cross-entropy: functional cases and N=10,000, C=1,000
./build/default/source/categorical_cross_entropy/categorical_cross_entropy
./build/default/source/categorical_cross_entropy/categorical_cross_entropy --performance

# Gaussian blur: functional cases and a 512x512 image with a 7x7 kernel
./build/default/source/gaussian_blur/gaussian_blur
./build/default/source/gaussian_blur/gaussian_blur --performance
```

Performance output separates CPU throughput, GPU kernel-only throughput, and GPU
end-to-end throughput. Host-side timing uses `benchmark()` and
`benchmarkWithReset()`; device-measured kernel time uses `benchmarkDevice()` and
`benchmarkDeviceWithReset()`. Both live in `source/utils/benchmark.h`. Shared unit
converters for GFLOP/s, GB/s, million items/s, and speedup operate on a result's
`median_ms` and live in `source/utils/benchmark_helpers.h`.

## Project Structure

```
gpu_playground/
├── source/                    ← all C++/CUDA/OpenCL source code
│   ├── utils/                 ← shared utility libraries
│   │   ├── cuda_helpers.h         ← CUDA_CHECK macro + getTime()
│   │   ├── benchmark.h            ← benchmark()/benchmarkWithReset(), benchmarkDevice()/benchmarkDeviceWithReset(), summarize()
│   │   ├── benchmark_helpers.h    ← unit converters (GFLOP/s, GB/s, items/s, speedup) over median_ms
│   │   ├── opencl_c_helpers.h     ← C API: CL_CHECK + clSetupGPU/clBuildFromSource/clTeardown
│   │   ├── opencl_helpers.h       ← C++ wrapper: clppGetGPUDevice/clppBuildProgram/clppPreferredLocalSize
│   │   └── python/                ← gpu_bench.py (torch-only benchmark()/print_bench()/check_close()/require()) + requirements.txt
│   ├── gemm/                  ← each kernel: main.cpp + cuda/ + opencl/ + opencl_cpp/
│   │   ├── main.cpp           ← backend-agnostic test harness (3-way #ifdef)
│   │   ├── cuda/              ← CUDA kernel + host-pointer wrapper
│   │   ├── opencl/            ← OpenCL C API implementation
│   │   └── opencl_cpp/        ← OpenCL C++ wrapper implementation
│   └── ...                    ← additional kernels; backend coverage is listed below
├── CMakeLists.txt             ← root build file; USE_OPENCL / USE_OPENCL_CPP + GPU_ENABLE_* flags
├── CMakePresets.json          ← build presets (default, native, ampere, release)
├── cuda_perf_analysis.sh      ← performance profiling script
├── docs/                      ← documentation index (docs/README.md) and guides
├── specs/                     ← dated feature packages for planned changes
├── gpu-mode/                  ← GPU MODE competition tools and submissions
└── .devcontainer/             ← Docker-based CUDA development environment
```

The `.devcontainer/` directory defines the Docker-based CUDA environment, and
`cuda_perf_analysis.sh` auto-detects `ncu`, `nsys`, or `nvprof` for profiling.

### Kernel Inventory

Portable kernels under `source/` provide `cuda/`, `opencl/`, and `opencl_cpp/`
implementations behind a shared `main.cpp` harness. The backends column records
the coverage actually built today.

| Kernel | Description | Backends |
|--------|-------------|----------|
| `gemm` | General matrix multiplication with tiled shared memory (`α·A·B + β·C`) | CUDA, OpenCL, OpenCL C++ |
| `sigmoid` | Sigmoid activation, scalar and vectorized `float4` variants | CUDA, OpenCL, OpenCL C++; Triton |
| `softmax` | Numerically stable softmax using a multi-stage reduction | CUDA, OpenCL, OpenCL C++ |
| `prefix_sum` | Parallel inclusive prefix scan (Hillis–Steele) | CUDA, OpenCL, OpenCL C++ |
| `geglu` | Gated linear unit with GELU activation, as used in Transformer FFNs | CUDA, OpenCL, OpenCL C++ |
| `silu` | SiLU / Swish activation (`x·σ(x)`) | CUDA, OpenCL, OpenCL C++ |
| `swiglu` | SwiGLU gated activation | CUDA, OpenCL, OpenCL C++ |
| `matrix_mul` | Basic tiled matrix multiplication | CUDA, OpenCL, OpenCL C++ |
| `matrix_transpose` | Matrix transpose using shared memory | CUDA, OpenCL, OpenCL C++ |
| `spmv` | Sparse matrix-vector multiplication, four kernel variants | CUDA, OpenCL, OpenCL C++ |
| `interleave_arrays` | Array interleaving (AoS ↔ SoA patterns) | CUDA, OpenCL, OpenCL C++ |
| `reverse_array` | In-place array reversal | CUDA, OpenCL, OpenCL C++ |
| `value_clipping` | Element-wise value clamping | CUDA, OpenCL, OpenCL C++ |
| `rgb_to_grayscale` | RGB to grayscale conversion using Rec.601 luminance | CUDA, OpenCL, OpenCL C++ |
| `convolution2d` | 2D convolution with tiled shared memory (CUDA) / naive (OpenCL) | CUDA, OpenCL, OpenCL C++ |
| `deep_learning_inference` | Small feed-forward CNN (RGBA to grayscale) with reference and candidate paths for optimization exercises | CUDA |
| `fp16_dot_product` | FP16 dot product using packed `__half2` loads, FP32 multiply-accumulate, FP16 result | CUDA |
| `categorical_cross_entropy` | Numerically stable categorical cross-entropy with block reductions for row maxima and exponential sums (LeetGPU challenge 25) | CUDA |
| `gaussian_blur` | Naive 2D Gaussian blur with zero padding; one thread per output pixel reads every tap from global memory (LeetGPU challenge 28) | CUDA |
| `vector_addition` | Element-wise vector addition | Triton, CuTe DSL |

`deep_learning_inference` needs generated weights before its first run: execute
`python source/deep_learning_inference/generate_weights.py`. It supports three
modes — `benchmark`, `image_infer <input> <output>`, and `correctness <image>` —
documented in [source/deep_learning_inference/README.md](source/deep_learning_inference/README.md).

## Python DSL Backends

Some kernels also ship optional Python DSL backends under
`source/<kernel>/python/<dsl>/<kernel>.py`, where `<dsl>` is `triton` or
`cute-dsl`. These are standalone scripts rather than CMake targets. A kernel may
be Python-only, as `vector_addition` is: it has no CUDA or OpenCL target.

Install the shared dependencies:

```bash
pip install -r source/utils/python/requirements.txt
```

Run the scripts from the repository root:

```bash
python source/sigmoid/python/triton/sigmoid.py
python source/vector_addition/python/triton/vector_addition.py --performance
python source/vector_addition/python/cute-dsl/vector_addition.py
```

Every script uses the shared `gpu_bench` helper and exits `0` for pass, `1` for
failure, or `77` when an optional dependency is missing (reported as SKIPPED).
See "Testing And Benchmarking" in [AGENTS.md](AGENTS.md) for the full contract.

## Documentation and Planned Work

- [docs/README.md](docs/README.md) is the documentation index: setup, guides,
  optimization rulebooks, and complexity studies.
- `specs/` holds dated feature packages (`requirements.md`, `plan.md`,
  `validation.md`) for planned or in-flight work. These describe intent, not
  shipped behavior.
- `lora_linear` is design-only and has no build target yet; see the
  [LoRA linear plan](specs/2026-07-14-lora-linear/plan.md).

## GPU MODE Competition

The `gpu-mode/` directory contains competition tooling and submissions:

- `SETUP_GUIDE.md` — setup for GPU MODE kernel competitions
- `tools/` — helper scripts for setup, submission, and quick reference
- `submissions/` — example submissions, including an NVFP4 GEMM baseline

## Examples

For framework comparisons and code examples, see
[docs/guides/examples.md](docs/guides/examples.md).

For agent workflows and repository-specific guidance, start with
[AGENTS.md](AGENTS.md) and the [documentation index](docs/README.md).

## Further Resources and References

### GPU Programming Communities

* [GPU Mode GitHub](https://github.com/gpu-mode)
* [LeetGPU](https://leetgpu.com)
* [reference-kernels](https://github.com/gpu-mode/reference-kernels) — reference kernels for the KernelBot competitions on [discord.gg/gpumode](https://discord.gg/gpumode)
* [Course on CUDA programming at Oxford Mathematical Institute](https://people.maths.ox.ac.uk/~gilesm/cuda/)
* [Interactive GPU learning resource](https://brrrviz.com/)

### Frameworks and Libraries

* [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
* [TinyGrad GitHub](https://github.com/geohot/tinygrad)
* [Triton GitHub](https://github.com/openai/triton)
* [CUTLASS GitHub](https://github.com/NVIDIA/cutlass) — high-performance GEMM templates for CUDA
* [NVIDIA CUDA Tile](https://developer.nvidia.com/cuda/tile) — cuTile Python programming language for GPUs
* [Mojo Documentation](https://docs.modular.com/mojo)

Setup instructions for these frameworks are in
[docs/setup/external-frameworks.md](docs/setup/external-frameworks.md).

### NVIDIA CCCL (CUDA C++ Core Libraries)

* [CCCL GitHub](https://github.com/NVIDIA/cccl) — unified repository containing:
  - **Thrust** — high-level C++ parallel algorithms
  - **CUB** — low-level CUDA building blocks and primitives
  - **libcu++** — the CUDA C++ standard library

### AI-Assisted CUDA Development

* [AI CUDA Engineer: Official Paper and Leaderboard](https://pub.sakana.ai/ai-cuda-engineer)
* [AI CUDA Engineer: Dataset](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)
* [ByteDance CUDA Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent)
