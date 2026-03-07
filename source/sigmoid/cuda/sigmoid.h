#pragma once
#include <cuda_runtime.h>

#define BSIZE 256

__global__ void sigmoid_kernel(const float* X, float* Y, int N);

__global__ void sigmoid_kernel2(const float* __restrict__ X, float* __restrict__ Y, int N);

void sigmoid_cpu(const float* input, float* output, int N);

void sigmoid(const float* d_input, float* d_output, int N);
void sigmoid2(const float* d_input, float* d_output, int N);

// Host-pointer wrappers (backend-agnostic API used by main.cpp)
void sigmoid_gpu(const float* h_input, float* h_output, int N);
void sigmoid2_gpu(const float* h_input, float* h_output, int N);
