#pragma once
#include <cuda_runtime.h>

#define BSIZE 256

__global__ void silu_kernel(const float* input, float* output, int N);

void silu_cpu(const float* input, float* output, int N);

void silu(const float* d_input, float* d_output, int N);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void silu_gpu(const float* h_input, float* h_output, int N);
