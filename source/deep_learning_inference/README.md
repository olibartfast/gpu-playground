# deep_learning_inference

This example is a CUDA-only RGBA-to-grayscale inference pipeline. It builds the
`deep_learning_inference` executable and runs a small feed-forward CNN using
weights generated from the bundled PyTorch definition.

The candidate path currently delegates to the reference implementation. This
provides a correctness-safe baseline for replacing individual stages with
optimized kernels while retaining the reference/candidate benchmark interface.

## Prerequisites

- Build the project with CUDA enabled.
- Install Python dependencies for weight generation:

```bash
pip install torch numpy
```

## 1. Generate weights

Run this once before the first execution:

```bash
python source/deep_learning_inference/generate_weights.py
```

That writes `source/deep_learning_inference/weights.bin`, which the executable
loads at runtime.

## 2. Configure and build

From the repository root:

```bash
cmake -S . -B build
cmake --build build --target deep_learning_inference -j
```

If you want to build only this example, you can also disable the other targets:

```bash
cmake -S . -B build -DGPU_ENABLE_DEEP_LEARNING_INFERENCE=ON
cmake --build build --target deep_learning_inference -j
```

Note: `USE_OPENCL` and `USE_OPENCL_CPP` do not support this example.

## 3. Run

All commands below are run from the repository root.

Benchmark mode:

```bash
./build/source/deep_learning_inference/deep_learning_inference benchmark
```

Correctness check against the reference implementation:

```bash
./build/source/deep_learning_inference/deep_learning_inference correctness images/image0.png
```

Run inference on an image and write the grayscale output:

```bash
./build/source/deep_learning_inference/deep_learning_inference image_infer images/image0.png out.png
```

## Input requirements

- Input images must be readable by `stb_image`.
- The program loads images as 4-channel RGBA.
- Image height and width must both be multiples of 128.
