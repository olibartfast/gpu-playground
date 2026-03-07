#include "swiglu.h"
#include "cuda_helpers.h"
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
    CUDA_CHECK(cudaGetLastError());
}

void swiglu_gpu(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * (N / 2)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice));
    swiglu(d_input, d_output, N);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float) * (N / 2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}
