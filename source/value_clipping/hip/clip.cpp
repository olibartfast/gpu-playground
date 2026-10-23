#include "clip.h"
#include "hip_helpers.h"

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
    HIP_CHECK(hipGetLastError());
}

void clip_gpu(const float* h_input, float* h_output, int N, float lo, float hi) {
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, sizeof(float) * N));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float) * N));
    HIP_CHECK(hipMemcpy(d_input, h_input, sizeof(float) * N, hipMemcpyHostToDevice));
    clip(d_input, d_output, N, lo, hi);
    HIP_CHECK(hipMemcpy(h_output, d_output, sizeof(float) * N, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}
