#pragma once

// 2D "same" convolution with zero padding (matches torch conv2d with
// padding = kernel/2). Odd kernel sizes are assumed.
void gaussian_blur_cpu(const float* input, const float* kernel, float* output,
                       int input_rows, int input_cols, int kernel_rows,
                       int kernel_cols);

// Copies host inputs to the device and the result back to the host.
// kernel_time_ms excludes allocation and transfers when non-null.
void gaussian_blur_gpu(const float* input, const float* kernel, float* output,
                       int input_rows, int input_cols, int kernel_rows,
                       int kernel_cols, float* kernel_time_ms = nullptr);

// LeetGPU-compatible entrypoint (challenge 28). All pointers are device pointers.
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols);
