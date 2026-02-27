// nvcc -x cu -o transpose transpose.cpp -std=c++17
// on tesla T4 GPU
// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o transpose transpose.cpp
#include "matrix_transpose.h"

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
