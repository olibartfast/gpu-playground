// nvcc -x cu -o reverse reverse.cpp -std=c++17
// on tesla T4 GPU
// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o reverse reverse.cpp
#include "reverse.h"

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
