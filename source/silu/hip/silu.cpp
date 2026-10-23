#include "silu.h"
#include "hip_helpers.h"
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
    HIP_CHECK(hipGetLastError());
}

void silu_gpu(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, sizeof(float) * N));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float) * N));
    HIP_CHECK(hipMemcpy(d_input, h_input, sizeof(float) * N, hipMemcpyHostToDevice));
    silu(d_input, d_output, N);
    HIP_CHECK(hipMemcpy(h_output, d_output, sizeof(float) * N, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}
