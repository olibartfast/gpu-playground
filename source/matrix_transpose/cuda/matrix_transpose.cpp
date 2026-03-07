// nvcc -x cu -o transpose transpose.cpp -std=c++17
// on tesla T4 GPU
// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o transpose transpose.cpp
#include "matrix_transpose.h"
#include "cuda_helpers.h"

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= rows || c >= cols)
        return;
    output[c*rows + r] = input[r*cols + c];
}

void matrix_transpose_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            output[c*rows + r] = input[r*cols + c];
        }
    }
}

void matrix_transpose_gpu(const float* h_input, float* h_output, int rows, int cols) {
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}
