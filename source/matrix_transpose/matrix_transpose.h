#pragma once
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols);

void matrix_transpose_cpu(const float* input, float* output, int rows, int cols);
