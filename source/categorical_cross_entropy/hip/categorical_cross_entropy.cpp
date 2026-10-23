#include "categorical_cross_entropy.h"

#include "hip_helpers.h"

#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace {

constexpr int BLOCK_SIZE = 256;
// Explicit logical subgroup width also works within AMD 64-lane wavefronts.
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down(value, offset, WARP_SIZE));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset, WARP_SIZE);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_max(float value, float *shared) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;

  value = warp_reduce_max(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  if (warp == 0) {
    value = warp_reduce_max(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float block_reduce_sum(float value, float *shared) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < NUM_WARPS) ? shared[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__global__ void
categorical_cross_entropy_kernel(const float *__restrict__ logits,
                                 const int *__restrict__ true_labels,
                                 float *__restrict__ loss, int n, int c) {
  const int sample = blockIdx.x;
  if (sample >= n) {
    return;
  }

  const int row_offset = sample * c;
  __shared__ float shared[NUM_WARPS];

  float thread_max = -FLT_MAX;
  for (int class_index = threadIdx.x; class_index < c;
       class_index += blockDim.x) {
    thread_max = fmaxf(thread_max, logits[row_offset + class_index]);
  }
  const float row_max = block_reduce_max(thread_max, shared);

  float thread_sum = 0.0f;
  for (int class_index = threadIdx.x; class_index < c;
       class_index += blockDim.x) {
    const float shifted_logit = logits[row_offset + class_index] - row_max;
    thread_sum += expf(shifted_logit);
  }
  const float exponential_sum = block_reduce_sum(thread_sum, shared);

  if (threadIdx.x == 0) {
    const int true_class = true_labels[sample];
    const float true_logit = logits[row_offset + true_class];
    const float sample_loss = row_max + logf(exponential_sum) - true_logit;
    atomicAdd(loss, sample_loss / static_cast<float>(n));
  }
}

} // namespace

extern "C" void solve(const float *logits, const int *true_labels, float *loss,
                      int n, int c) {
  HIP_CHECK(hipMemset(loss, 0, sizeof(float)));
  categorical_cross_entropy_kernel<<<n, BLOCK_SIZE>>>(logits, true_labels, loss,
                                                      n, c);
  HIP_CHECK(hipGetLastError());
}

float categorical_cross_entropy_cpu(const float *logits, const int *true_labels,
                                    int n, int c) {
  double total_loss = 0.0;
  for (int sample = 0; sample < n; ++sample) {
    const int row_offset = sample * c;
    float row_max = -std::numeric_limits<float>::max();
    for (int class_index = 0; class_index < c; ++class_index) {
      row_max = std::fmax(row_max, logits[row_offset + class_index]);
    }

    float exponential_sum = 0.0f;
    for (int class_index = 0; class_index < c; ++class_index) {
      exponential_sum += std::exp(logits[row_offset + class_index] - row_max);
    }
    const float sample_loss = row_max + std::log(exponential_sum) -
                              logits[row_offset + true_labels[sample]];
    total_loss += static_cast<double>(sample_loss);
  }
  return static_cast<float>(total_loss / static_cast<double>(n));
}

float categorical_cross_entropy_gpu(const float *h_logits,
                                    const int *h_true_labels, int n, int c,
                                    float *kernel_time_ms) {
  if (n <= 0 || c <= 0) {
    if (kernel_time_ms != nullptr) {
      *kernel_time_ms = 0.0f;
    }
    return 0.0f;
  }

  const std::size_t logits_bytes =
      sizeof(float) * static_cast<std::size_t>(n) * c;
  const std::size_t labels_bytes = sizeof(int) * static_cast<std::size_t>(n);
  float *d_logits = nullptr;
  int *d_true_labels = nullptr;
  float *d_loss = nullptr;
  HIP_CHECK(hipMalloc(&d_logits, logits_bytes));
  HIP_CHECK(hipMalloc(&d_true_labels, labels_bytes));
  HIP_CHECK(hipMalloc(&d_loss, sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_logits, h_logits, logits_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_true_labels, h_true_labels, labels_bytes,
                        hipMemcpyHostToDevice));

  hipEvent_t kernel_start;
  hipEvent_t kernel_end;
  HIP_CHECK(hipEventCreate(&kernel_start));
  HIP_CHECK(hipEventCreate(&kernel_end));
  HIP_CHECK(hipEventRecord(kernel_start));
  solve(d_logits, d_true_labels, d_loss, n, c);
  HIP_CHECK(hipEventRecord(kernel_end));
  HIP_CHECK(hipEventSynchronize(kernel_end));
  if (kernel_time_ms != nullptr) {
    HIP_CHECK(hipEventElapsedTime(kernel_time_ms, kernel_start, kernel_end));
  }
  HIP_CHECK(hipEventDestroy(kernel_start));
  HIP_CHECK(hipEventDestroy(kernel_end));

  float result = 0.0f;
  HIP_CHECK(
      hipMemcpy(&result, d_loss, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_logits));
  HIP_CHECK(hipFree(d_true_labels));
  HIP_CHECK(hipFree(d_loss));
  return result;
}
