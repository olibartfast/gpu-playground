#pragma once
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_pixels);

void rgb_to_grayscale_cpu(const float* input, float* output, int total_pixels);
