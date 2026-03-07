/**
 * @brief Computes the GEGLU (Gated Gaussian Error Linear Unit).
 *
 * GEGLU is a Gated Linear Unit variant that replaces standard point-wise
 * activations (like ReLU or GELU) in Transformer Feed-Forward networks.
 * It has been shown to offer better convergence and performance than standard
 * GELU in Large Language Models (e.g., PaLM, T5).
 *
 * Mechanism:
 *   Unlike standard activations, GEGLU requires splitting the input dimension
 *   (or taking two separate input projections).
 *   1. Split input x into two halves: x1 (the value) and x2 (the gate).
 *   2. Apply GELU activation to x2.
 *   3. Multiply x1 by the activated x2.
 *
 * Formula:
 *     GEGLU(x1, x2) = x1 * GELU(x2)
 *     GEGLU(x1, x2) = x1 * (0.5 * x2 * (1 + erf(x2 / sqrt(2))))
 */

#include "geglu.h"
#include "cuda_helpers.h"
#include <cmath>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= halfN)
        return;
    float x1 = input[i];
    float x2 = input[i + halfN];
    float gelu_x2 = 0.5f * x2 * (1.0f + std::erf(x2 * 0.70710678f));
    output[i] = x1 * gelu_x2;
}

void geglu_cpu(const float* input, float* output, int halfN) {
    for(size_t i = 0; i < halfN; i++)
    {
        float x1 = input[i];
        float x2 = input[i + halfN];
        float gelu_x2 = 0.5f * x2 * (1.0f + std::erf(x2 * 0.70710678f));
        output[i] = x1 * gelu_x2;
    }
}

void geglu(const float* input, float* output, int halfN) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void geglu_gpu(const float* h_input, float* h_output, int halfN) {
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * 2 * halfN));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * halfN));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(float) * 2 * halfN, cudaMemcpyHostToDevice));
    geglu(d_input, d_output, halfN);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float) * halfN, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}
