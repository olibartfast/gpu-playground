#pragma once
#include <cuda_runtime.h>

void convolution2d_cpu(const float* input, const float* kernel, float* output,
                       int input_rows, int input_cols,
                       int kernel_rows, int kernel_cols);

// Tiled: loads input patch into shared memory
void convolution2d_gpu(const float* h_input, const float* h_kernel, float* h_output,
                       int input_rows, int input_cols,
                       int kernel_rows, int kernel_cols);

// Fused: loads kernel weights into shared memory (DALI im2col pattern)
void convolution2d_gpu2(const float* h_input, const float* h_kernel, float* h_output,
                        int input_rows, int input_cols,
                        int kernel_rows, int kernel_cols);
