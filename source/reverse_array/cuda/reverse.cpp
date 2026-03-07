// nvcc -x cu -o reverse reverse.cpp -std=c++17
// on tesla T4 GPU
// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o reverse reverse.cpp
#include "reverse.h"
#include "cuda_helpers.h"

__global__ void reverse_array(float* input, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int j = N - tid - 1;
    if (tid < N/2) {
        float tmp = input[tid];
        input[tid] = input[j];
        input[j] = tmp;
    }
}

void reverse_array_cpu(float* input, int N) {
    int i = 0, j = N - 1;
    while (i < j) {
        float tmp = input[i];
        input[i] = input[j];
        input[j] = tmp;
        i++;
        j--;
    }
}

void reverse_array_gpu(float* h_input, int N) {
    float* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice));
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_input, d_input, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
}
