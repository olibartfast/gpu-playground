#pragma once
#include <cuda_runtime.h>

#define BSIZE 256

__global__ void geglu_kernel(const float* input, float* output, int halfN);

void geglu_cpu(const float* input, float* output, int halfN);

void geglu(const float* input, float* output, int halfN);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void geglu_gpu(const float* h_input, float* h_output, int halfN);
