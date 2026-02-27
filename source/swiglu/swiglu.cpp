// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o swiglu swiglu.cpp
#include "swiglu.h"
#include <iostream>
#include <cmath>

__global__ void swiglu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N/2) {
        float x1 = input[i];
        float x2 = input[i + N/2];
        float silu = x1 / (1.0f + expf(-x1));
        output[i] = silu * x2;
    }
}

void swiglu_cpu(const float* input, float* output, int N) {
    // SwiGLU combines gating and activation
    // First half is processed through SiLU, second half is gating
    for (int i = 0; i < N/2; i++) {
        float x1 = input[i];
        float x2 = input[i + N/2];
        // SiLU activation: x * sigmoid(x)
        float silu = x1 / (1.0f + expf(-x1));
        // Multiply with gating value
        output[i] = silu * x2;
    }
}

void swiglu(const float* d_input, float* d_output, int N) {
    int threadsPerBlock = BSIZE;
    // We only need half as many threads since each thread processes two elements
    int blocksPerGrid = ((N/2) + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
