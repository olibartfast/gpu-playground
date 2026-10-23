#pragma once
#include <hip/hip_runtime.h>

#define BSIZE 256

__global__ void gelu_kernel(const float* X, float* Y, int N);

void gelu_cpu(const float* X, float* Y, int N);

void gelu(const float* X, float* Y, int N);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void gelu_gpu(const float* h_input, float* h_output, int N);
