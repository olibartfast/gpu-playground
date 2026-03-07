#include "clip.h"
#include "cuda_helpers.h"

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
    CUDA_CHECK(cudaGetLastError());
}

void clip_gpu(const float* h_input, float* h_output, int N, float lo, float hi) {
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice));
    clip(d_input, d_output, N, lo, hi);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}
