#include "silu.h"
#include "cuda_helpers.h"
#include <cmath>

// Sigmoid Linear Unit / Swish
/**
 * @brief Computes the SiLU (Sigmoid Linear Unit) activation, also known as Swish.
 *
 * SiLU is a smooth, non-monotonic activation function often used in modern
 * deep learning architectures (e.g., EfficientNet, YOLO) as an alternative to ReLU.
 *
 * Formula:
 *     f(x) = x * sigmoid(x)
 *     f(x) = x / (1.0 + exp(-x))
 */

__global__ void silu_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= N )
        return;
    float sigma = 1.0f / (1 + exp(-input[i]));
    output[i] = input[i]*sigma;
}

void silu_cpu(const float* input, float* output, int N) {
    for(int i=0; i<N; i++) {
        float sigma = 1.0f / (1 + exp(-input[i]));
        output[i] = input[i]*sigma;
    }
}

void silu(const float* d_input, float* d_output, int N) {
    int threadsPerBlock = BSIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CUDA_CHECK(cudaGetLastError());
}

void silu_gpu(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice));
    silu(d_input, d_output, N);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}
