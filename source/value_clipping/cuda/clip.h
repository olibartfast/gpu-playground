#pragma once
#include <cuda_runtime.h>

#define BSIZE 256

__global__ void clip_kernel(const float* input, float* output, int N, float lo, float hi);

void clip_cpu(const float* input, float* output, int N, float lo, float hi);

void clip(const float* d_input, float* d_output, int N, float lo, float hi);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void clip_gpu(const float* h_input, float* h_output, int N, float lo, float hi);
