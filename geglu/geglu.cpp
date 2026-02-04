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
 *
 * Characteristics:
 *   - Gating: x2 acts as a "soft switch" that regulates the flow of x1.
 *   - Quadratic: Because it multiplies two linear projections, it introduces
 *     a quadratic dependence on the input.
 *   - Dimensionality: Requires the preceding linear layer to project to
 *     2x the hidden size (to allow for the split) to maintain width.
 *
 * @param x1 The first half of the split input (the pass-through value).
 * @param x2 The second half of the split input (the gate input).
 * @return The gated value.
 */
// float geglu(float x1, float x2) {
//     float gelu_x2 = 0.5f * x2 * (1.0f + std::erf(x2 * 0.70710678f));
//     return x1 * gelu_x2;
// }

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <vector>



#define BSIZE 256

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >=halfN) 
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
    cudaDeviceSynchronize();
}


int main(int argc, char const *argv[])
{
    int N = 1024;
    std::vector<float> input(N);
    std::vector<float> output_cpu(N);
    std::vector<float> output_gpu(N);
    for(int i=0; i<N; i++) {
        input[i] = (float)(i % 100);
    }

    auto start_cpu = std::chrono::steady_clock::now();
    geglu_cpu(input.data(), output_cpu.data(), N/2);
    auto end_cpu = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU duration: " << duration_cpu.count() << " ms" << std::endl;

    // GPU computation
    auto start_gpu = std::chrono::steady_clock::now();
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);
    cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    geglu(d_input, d_output, N/2);
    cudaMemcpy(output_gpu.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    auto end_gpu = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration_gpu = end_gpu - start_gpu;
    std::cout << "GPU duration: " << duration_gpu.count() << " ms" << std::endl;
    cudaFree(d_input);
    cudaFree(d_output); 
    for(int i=0; i<N; i++) {
        if(fabs(output_cpu[i] - output_gpu[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": CPU " << output_cpu[i] << " vs GPU " << output_gpu[i] << std::endl;
            return -1;
        }
    }
    std::cout << "Results match!" << std::endl;
    return 0;
}


