# Deep Learning Inference

A CUDA-only RGBA-to-grayscale inference pipeline. It builds the
`deep_learning_inference` executable, which runs a small feed-forward CNN using
weights generated from the bundled PyTorch definition.

The candidate path currently delegates to the reference implementation. That
gives you a correctness-safe baseline for replacing individual stages with
optimized kernels while keeping the reference-versus-candidate benchmark
interface.

## Prerequisites

- Build the project with CUDA enabled.
- Install the Python dependencies for weight generation:

```bash
pip install torch numpy
```

## 1. Generate Weights

Run this once before the first execution:

```bash
python source/deep_learning_inference/generate_weights.py
```

The script writes `source/deep_learning_inference/weights.bin`, which the
executable loads at runtime.

## 2. Configure and Build

From the repository root:

```bash
cmake -S . -B build
cmake --build build --target deep_learning_inference -j
```

To build only this example, disable the other targets as well:

```bash
cmake -S . -B build -DGPU_ENABLE_DEEP_LEARNING_INFERENCE=ON
cmake --build build --target deep_learning_inference -j
```

`USE_OPENCL` and `USE_OPENCL_CPP` do not support this example.

## 3. Run

All commands below run from the repository root.

Benchmark mode:

```bash
./build/source/deep_learning_inference/deep_learning_inference benchmark
```

Correctness check against the reference implementation:

```bash
./build/source/deep_learning_inference/deep_learning_inference correctness images/image0.png
```

Inference on an image, writing the grayscale output:

```bash
./build/source/deep_learning_inference/deep_learning_inference image_infer images/image0.png out.png
```

## Input Requirements

- Input images must be readable by `stb_image`.
- The program loads images as four-channel RGBA.
- Image height and width must both be multiples of 128.
