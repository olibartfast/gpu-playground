#pragma once
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N);

void reverse_array_cpu(float* input, int N);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void reverse_array_gpu(float* h_input, int N);
