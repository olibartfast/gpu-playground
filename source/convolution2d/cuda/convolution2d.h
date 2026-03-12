#pragma once
#include <cuda_runtime.h>

void convolution2d_cpu(const float* input, const float* kernel, float* output,
                       int input_rows, int input_cols,
                       int kernel_rows, int kernel_cols);

void convolution2d_gpu(const float* h_input, const float* h_kernel, float* h_output,
                       int input_rows, int input_cols,
                       int kernel_rows, int kernel_cols);
