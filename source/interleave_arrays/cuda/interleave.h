#pragma once
#include <cuda_runtime.h>

#define BSIZE 256

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N);

void interleave_cpu(const float* A, const float* B, float* output, int N);

void interleave(const float* d_A, const float* d_B, float* d_output, int N);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void interleave_gpu(const float* h_A, const float* h_B, float* h_output, int N);
