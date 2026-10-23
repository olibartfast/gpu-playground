/**
 * @brief Computes the GELU (Gaussian Error Linear Unit) activation.
 *
 * Formula:
 *     GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * Element-wise: output has the same size as input.
 */

#include "gelu.h"
#include "hip_helpers.h"
#include <cmath>

constexpr float INV_SQRT2 = 0.70710678118654752f; // 1 / sqrt(2)

__global__ void gelu_kernel(const float* X, float* Y, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N)
        return;
    float x = X[i];
    Y[i] = 0.5f * x * (1.0f + erff(x * INV_SQRT2));
}

void gelu_cpu(const float* X, float* Y, int N)
{
    for (int i = 0; i < N; i++)
    {
        float x = X[i];
        Y[i] = 0.5f * x * (1.0f + std::erf(x * INV_SQRT2));
    }
}

// d_input / d_output are device pointers
void gelu(const float* input, float* output, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

// h_input / h_output are host pointers
void gelu_gpu(const float* h_input, float* h_output, int N)
{
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, sizeof(float) * N));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float) * N));
    HIP_CHECK(hipMemcpy(d_input, h_input, sizeof(float) * N, hipMemcpyHostToDevice));
    gelu(d_input, d_output, N);
    HIP_CHECK(hipMemcpy(h_output, d_output, sizeof(float) * N, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}