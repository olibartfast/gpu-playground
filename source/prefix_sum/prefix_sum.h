#pragma once
#include <cuda_runtime.h>

void prefix_scan_cpu(float* input, float* output, int N);

__global__ void block_inclusive_scan(const float *input, float *output, float *block_sums, int n, int block_size);

__global__ void scan_block_sums(float *block_sums, int num_blocks);

__global__ void add_block_sums(float *output, float *block_sums, int n, int block_size);
