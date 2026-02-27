/**
 * Sigmoid Activation Function - CUDA Implementation
 *
 * The sigmoid function maps any real number to the range (0, 1):
 *
 *     σ(x) = 1 / (1 + e^(-x))
 *
 * Characteristics:
 *   - Output range: (0, 1), making it useful for binary classification outputs
 *     and gating mechanisms.
 *   - Saturates at 0 for large negative inputs and at 1 for large positive inputs,
 *     which can cause vanishing gradients during backpropagation.
 *   - Derivative: σ'(x) = σ(x) * (1 - σ(x)), which peaks at 0.25 when x = 0.
 *
 * Two kernel implementations are provided:
 *   - sigmoid_kernel:  scalar, one element per thread.
 *   - sigmoid_kernel2: vectorized, four elements per thread via float4
 *                      loads/stores and the __expf() fast-math intrinsic.
 */

#include "sigmoid.h"
#include <iostream>
#include <cmath>

__global__ void sigmoid_kernel(const float* X, float* Y, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= N)
        return;
    Y[i] = 1.0f / (1 + exp(-X[i]));
}

// Vectorized kernel: processes 4 elements per thread via float4 loads/stores.
// Uses __expf() fast-math intrinsic and __restrict__ to signal no aliasing.
__global__ void sigmoid_kernel2(const float* __restrict__ X, float* __restrict__ Y, int N)
{
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    if (i + 3 < N) {
        float4 x = reinterpret_cast<const float4*>(X)[i / 4];
        float4 y;
        y.x = 1.0f / (1.0f + __expf(-x.x));
        y.y = 1.0f / (1.0f + __expf(-x.y));
        y.z = 1.0f / (1.0f + __expf(-x.z));
        y.w = 1.0f / (1.0f + __expf(-x.w));
        reinterpret_cast<float4*>(Y)[i / 4] = y;
    } else {
        for (int j = i; j < N; j++) {
            Y[j] = 1.0f / (1.0f + __expf(-X[j]));
        }
    }
}

void sigmoid_cpu(const float* input, float* output, int N) {
    for(int i = 0; i < N; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

void sigmoid(const float* d_input, float* d_output, int N) {
    int threadsPerBlock = BSIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void sigmoid2(const float* d_input, float* d_output, int N) {
    int threadsPerBlock = BSIZE;
    // Each thread handles 4 elements
    int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel2<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
