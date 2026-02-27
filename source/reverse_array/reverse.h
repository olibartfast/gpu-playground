#pragma once
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N);

void reverse_array_cpu(float* input, int N);
