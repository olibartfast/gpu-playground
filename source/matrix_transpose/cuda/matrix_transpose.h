#pragma once
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols);

void matrix_transpose_cpu(const float* input, float* output, int rows, int cols);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void matrix_transpose_gpu(const float* h_input, float* h_output, int rows, int cols);
