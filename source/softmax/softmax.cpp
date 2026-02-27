// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o softmax softmax.cpp
#include "softmax.h"
#include <cmath>
#include <float.h>

void softmax_cpu(const float* input, float* output, int N) {
    // Step 1: Find maximum value for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, input[i]);
    }

    // Step 2: Compute sum of exponentials with max subtracted
    float sum_exp = 0.0f;
    for (int i = 0; i < N; i++) {
        sum_exp += expf(input[i] - max_val);
    }

    // Step 3: Compute softmax for each element
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
}

__global__ void find_max_partial_kernel(const float* input, float* partial_maxs, int N) {
    __shared__ float sdata[BSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    float my_max = -FLT_MAX;
    while (i < N) {
        my_max = fmaxf(my_max, input[i]);
        i += gridDim.x * blockDim.x;
    }
    sdata[tid] = my_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) partial_maxs[blockIdx.x] = sdata[0];
}

__global__ void sum_exp_partial_kernel(const float* input, float* partial_sums, const float* max_val, int N) {
    __shared__ float sdata[BSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    float my_sum = 0.0f;
    while (i < N) {
        my_sum += expf(input[i] - (*max_val));
        i += gridDim.x * blockDim.x;
    }
    sdata[tid] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

__global__ void reduce_final_kernel(float* partials, float* result, int N, bool is_max_reduction) {
    __shared__ float sdata[BSIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    float my_val = is_max_reduction ? -FLT_MAX : 0.0f;
    while (i < N) {
        if (is_max_reduction) my_val = fmaxf(my_val, partials[i]);
        else my_val += partials[i];
        i += blockDim.x;
    }
    sdata[tid] = my_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (is_max_reduction) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            else sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) result[0] = sdata[0];
}

__global__ void softmax_elementwise_kernel(const float* input, float* output, const float* max_val, const float* sum_exp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = expf(input[i] - (*max_val)) / (*sum_exp);
    }
}

// Host-side orchestrator for the efficient GPU softmax
void softmax_gpu_efficient(const float* d_input, float* d_output, int N) {
    int threadsPerBlock = BSIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_partials, *d_max_val, *d_sum_exp;
    cudaMalloc(&d_partials, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_max_val, sizeof(float));
    cudaMalloc(&d_sum_exp, sizeof(float));

    // Stage 1: Find max value
    find_max_partial_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_partials, N);
    reduce_final_kernel<<<1, threadsPerBlock>>>(d_partials, d_max_val, blocksPerGrid, true);

    // Stage 2: Sum exponentials
    sum_exp_partial_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_partials, d_max_val, N);
    reduce_final_kernel<<<1, threadsPerBlock>>>(d_partials, d_sum_exp, blocksPerGrid, false);

    // Stage 3: Final element-wise calculation
    softmax_elementwise_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_max_val, d_sum_exp, N);

    cudaFree(d_partials);
    cudaFree(d_max_val);
    cudaFree(d_sum_exp);
}
