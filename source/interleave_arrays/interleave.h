#pragma once
#include <cuda_runtime.h>

#define BSIZE 256

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N);

void interleave_cpu(const float* A, const float* B, float* output, int N);

void interleave(const float* d_A, const float* d_B, float* d_output, int N);
