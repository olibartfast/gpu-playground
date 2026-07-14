#include "fp16_dot_product.h"

#include "cuda_helpers.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_BLOCKS = 256;

__global__ void fp16_dot_partial_kernel(const __half* a,
                                        const __half* b,
                                        float* partial_sums,
                                        int n) {
    __shared__ float sums[THREADS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    const int packed_count = n / 2;
    float sum = 0.0f;

    const __half2* a2 = reinterpret_cast<const __half2*>(a);
    const __half2* b2 = reinterpret_cast<const __half2*>(b);
    for (int i = global_tid; i < packed_count; i += stride) {
        const float2 a_values = __half22float2(a2[i]);
        const float2 b_values = __half22float2(b2[i]);
        sum += a_values.x * b_values.x + a_values.y * b_values.y;
    }

    if (global_tid == 0 && (n & 1) != 0) {
        sum += __half2float(a[n - 1]) * __half2float(b[n - 1]);
    }

    sums[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sums[tid] += sums[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sums[0];
    }
}

__global__ void sum_partial_results_kernel(const float* partial_sums,
                                           __half* result,
                                           int count) {
    __shared__ float sums[THREADS_PER_BLOCK];

    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < count; i += blockDim.x) {
        sum += partial_sums[i];
    }
    sums[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sums[tid] += sums[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = __float2half(sums[0]);
    }
}

}  // namespace

float fp16_dot_product_cpu(const float* a, const float* b, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; ++i) {
        const __half a_fp16 = __float2half(a[i]);
        const __half b_fp16 = __float2half(b[i]);
        result += __half2float(a_fp16) * __half2float(b_fp16);
    }
    return __half2float(__float2half(result));
}

float fp16_dot_product_gpu(const float* h_a,
                           const float* h_b,
                           int n,
                           float* kernel_time_ms) {
    if (n <= 0) {
        if (kernel_time_ms != nullptr) {
            *kernel_time_ms = 0.0f;
        }
        return 0.0f;
    }

    std::vector<__half> h_a_fp16(static_cast<std::size_t>(n));
    std::vector<__half> h_b_fp16(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        h_a_fp16[i] = __float2half(h_a[i]);
        h_b_fp16[i] = __float2half(h_b[i]);
    }

    __half* d_a = nullptr;
    __half* d_b = nullptr;
    float* d_partial_sums = nullptr;
    __half* d_result = nullptr;

    const int packed_count = (n + 1) / 2;
    const int blocks = std::min(
        MAX_BLOCKS, (packed_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(__half) * n));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(__half) * n));
    CUDA_CHECK(cudaMalloc(&d_partial_sums, sizeof(float) * blocks));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(
        d_a, h_a_fp16.data(), sizeof(__half) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_b, h_b_fp16.data(), sizeof(__half) * n, cudaMemcpyHostToDevice));

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_end;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_end));
    CUDA_CHECK(cudaEventRecord(kernel_start));

    fp16_dot_partial_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_a, d_b, d_partial_sums, n);
    CUDA_CHECK(cudaGetLastError());
    sum_partial_results_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_partial_sums, d_result, blocks);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(kernel_end));
    CUDA_CHECK(cudaEventSynchronize(kernel_end));
    if (kernel_time_ms != nullptr) {
        CUDA_CHECK(cudaEventElapsedTime(kernel_time_ms, kernel_start, kernel_end));
    }
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_end));

    __half result_fp16;
    CUDA_CHECK(cudaMemcpy(
        &result_fp16, d_result, sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_partial_sums));
    CUDA_CHECK(cudaFree(d_result));
    return __half2float(result_fp16);
}
