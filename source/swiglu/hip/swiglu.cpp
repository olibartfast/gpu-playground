#include "swiglu.h"
#include "hip_helpers.h"
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
    HIP_CHECK(hipGetLastError());
}

void swiglu_gpu(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, sizeof(float) * N));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float) * (N / 2)));
    HIP_CHECK(hipMemcpy(d_input, h_input, sizeof(float) * N, hipMemcpyHostToDevice));
    swiglu(d_input, d_output, N);
    HIP_CHECK(hipMemcpy(h_output, d_output, sizeof(float) * (N / 2), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}
