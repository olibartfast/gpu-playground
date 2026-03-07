#include "interleave.h"
#include "cuda_helpers.h"

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Each thread handles one element from A and one from B
        // Interleaving pattern: [A[0], B[0], A[1], B[1], A[2], B[2], ...]
        output[2 * idx] = A[idx];
        output[2 * idx + 1] = B[idx];
    }
}

void interleave_cpu(const float* A, const float* B, float* output, int N) {
    // Interleave arrays A and B element by element
    for (int i = 0; i < N; i++) {
        output[2 * i] = A[i];
        output[2 * i + 1] = B[i];
    }
}

void interleave(const float* d_A, const float* d_B, float* d_output, int N) {
    int threadsPerBlock = BSIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_output, N);
    CUDA_CHECK(cudaGetLastError());
}

void interleave_gpu(const float* h_A, const float* h_B, float* h_output, int N) {
    float *d_A, *d_B, *d_output;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * 2 * N));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice));
    interleave(d_A, d_B, d_output, N);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float) * 2 * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_output));
}
