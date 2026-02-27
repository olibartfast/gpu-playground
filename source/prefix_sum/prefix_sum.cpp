// nvcc -x cu -arch=compute_75 -code=sm_75 -std=c++17 -o prefix_scan prefix_scan.cpp
#include "prefix_sum.h"

// CPU inclusive prefix scan
void prefix_scan_cpu(float* input, float* output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// Kernel 1: Per-block inclusive scan (Hillis-Steele)
__global__ void block_inclusive_scan(const float *input, float *output, float *block_sums, int n, int block_size) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * block_size;

    // Load input chunk into shared memory
    if (tid < block_size && block_offset + tid < n) {
        temp[tid] = input[block_offset + tid];
    } else {
        temp[tid] = 0; // Pad with identity
    }
    __syncthreads();

    // Hillis-Steele scan
    for (int offset = 1; offset < n; offset *= 2) {
        float t = 0;
        if (tid >= offset) {
            t = temp[tid - offset];
        }
        __syncthreads();
        if (tid < n && tid >= offset) {
            temp[tid] += t;
        }
        __syncthreads();
    }

    // Store block sum for multi-block case
    if (tid == 0 && block_offset + block_size - 1 < n) {
        block_sums[blockIdx.x] = temp[block_size - 1];
    }

    // Write to output
    if (tid < block_size && block_offset + tid < n) {
        output[block_offset + tid] = temp[tid];
    }
}

// Kernel 2: Scan block sums (Hillis-Steele)
__global__ void scan_block_sums(float *block_sums, int num_blocks) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;

    if (tid < num_blocks) {
        temp[tid] = block_sums[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < num_blocks; offset *= 2) {
        float t = 0;
        if (tid >= offset) {
            t = temp[tid - offset];
        }
        __syncthreads();
        if (tid < num_blocks && tid >= offset) {
            temp[tid] += t;
        }
        __syncthreads();
    }

    if (tid < num_blocks) {
        block_sums[tid] = temp[tid];
    }
}

// Kernel 3: Add block sums to per-block scans
__global__ void add_block_sums(float *output, float *block_sums, int n, int block_size) {
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * block_size;

    if (blockIdx.x > 0 && tid < block_size && block_offset + tid < n) {
        output[block_offset + tid] += block_sums[blockIdx.x - 1];
    }
}
