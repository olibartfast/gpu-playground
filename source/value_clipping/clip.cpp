// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o clip clip.cpp
#include "clip.h"
#include <iostream>

__global__ void clip_kernel(const float* input, float* output, int N, float lo, float hi) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = input[idx];
        // Clamp value to [lo, hi] range
        output[idx] = fminf(fmaxf(val, lo), hi);
    }
}

void clip_cpu(const float* input, float* output, int N, float lo, float hi) {
    for (int i = 0; i < N; i++) {
        float val = input[i];
        if (val < lo) {
            output[i] = lo;
        } else if (val > hi) {
            output[i] = hi;
        } else {
            output[i] = val;
        }
    }
}

void clip(const float* d_input, float* d_output, int N, float lo, float hi) {
    int threadsPerBlock = BSIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, lo, hi);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
