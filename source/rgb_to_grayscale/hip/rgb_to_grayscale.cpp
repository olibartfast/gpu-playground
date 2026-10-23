// nvcc -x cu -o rgb2gray rgb2gray.cpp -std=c++17
// on Tesla T4 GPU
// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o rgb2gray rgb2gray.cpp
#include "rgb_to_grayscale.h"
#include "hip_helpers.h"

__global__ void rgb_to_grayscale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_pixels; i += stride) {
        int input_idx = i * 3;
        float r = input[input_idx];
        float g = input[input_idx + 1];
        float b = input[input_idx + 2];
        output[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void rgb_to_grayscale_cpu(const float* input, float* output, int total_pixels) {
    for (int i = 0; i < total_pixels; i++) {
        int idx = i * 3;
        output[i] = 0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
    }
}

void rgb_to_grayscale_gpu(const float* h_input, float* h_output, int total_pixels) {
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, sizeof(float) * total_pixels * 3));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float) * total_pixels));
    HIP_CHECK(hipMemcpy(d_input, h_input, sizeof(float) * total_pixels * 3, hipMemcpyHostToDevice));
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, total_pixels);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h_output, d_output, sizeof(float) * total_pixels, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}
