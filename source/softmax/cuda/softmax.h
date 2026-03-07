#pragma once
#include <cuda_runtime.h>

#define BSIZE 256

void softmax_cpu(const float* input, float* output, int N);

__global__ void find_max_partial_kernel(const float* input, float* partial_maxs, int N);

__global__ void sum_exp_partial_kernel(const float* input, float* partial_sums, const float* max_val, int N);

__global__ void reduce_final_kernel(float* partials, float* result, int N, bool is_max_reduction);

__global__ void softmax_elementwise_kernel(const float* input, float* output, const float* max_val, const float* sum_exp, int N);

void softmax_gpu_efficient(const float* d_input, float* d_output, int N);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void softmax_gpu(const float* h_input, float* h_output, int N);
