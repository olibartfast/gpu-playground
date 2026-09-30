// Naive Gaussian blur: one thread per output pixel, every tap read from global memory.
#include "gaussian_blur.h"
#include "cuda_helpers.h"

#include <cstddef>

__global__ void gaussian_blur(const float *input, const float *kernel,
                              float *output, int input_rows, int input_cols,
                              int kernel_rows, int kernel_cols) {
  const int i = threadIdx.y + blockDim.y * blockIdx.y;
  const int j = threadIdx.x + blockDim.x * blockIdx.x;

  if (i >= input_rows || j >= input_cols)
    return;

  const int kr = kernel_rows / 2;
  const int kc = kernel_cols / 2;
  float sum = 0.0f;
  for (int u = -kr; u <= kr; ++u) {
    const int y = i + u;
    if (y < 0 || y >= input_rows)
      continue; // zero padding
    for (int v = -kc; v <= kc; ++v) {
      const int x = j + v;
      if (x < 0 || x >= input_cols)
        continue;
      sum += input[y * input_cols + x] *
             kernel[(u + kr) * kernel_cols + (v + kc)];
    }
  }
  output[i * input_cols + j] = sum;
}

extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  const int threadsPerBlock = 16;
  dim3 grid((input_cols + threadsPerBlock - 1) / threadsPerBlock,
            (input_rows + threadsPerBlock - 1) / threadsPerBlock);
  dim3 block(threadsPerBlock, threadsPerBlock);

  gaussian_blur<<<grid, block>>>(input, kernel, output, input_rows, input_cols,
                                 kernel_rows, kernel_cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void gaussian_blur_cpu(const float *input, const float *kernel, float *output,
                       int input_rows, int input_cols, int kernel_rows,
                       int kernel_cols) {
  const int kr = kernel_rows / 2;
  const int kc = kernel_cols / 2;
  for (int i = 0; i < input_rows; ++i) {
    for (int j = 0; j < input_cols; ++j) {
      float sum = 0.0f;
      for (int u = -kr; u <= kr; ++u) {
        const int y = i + u;
        if (y < 0 || y >= input_rows)
          continue;
        for (int v = -kc; v <= kc; ++v) {
          const int x = j + v;
          if (x < 0 || x >= input_cols)
            continue;
          sum += input[y * input_cols + x] *
                 kernel[(u + kr) * kernel_cols + (v + kc)];
        }
      }
      output[i * input_cols + j] = sum;
    }
  }
}

void gaussian_blur_gpu(const float *h_input, const float *h_kernel,
                       float *h_output, int input_rows, int input_cols,
                       int kernel_rows, int kernel_cols,
                       float *kernel_time_ms) {
  const std::size_t image_bytes =
      sizeof(float) * static_cast<std::size_t>(input_rows) * input_cols;
  const std::size_t kernel_bytes =
      sizeof(float) * static_cast<std::size_t>(kernel_rows) * kernel_cols;
  float *d_input = nullptr;
  float *d_kernel = nullptr;
  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
  CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));

  cudaEvent_t kernel_start;
  cudaEvent_t kernel_end;
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_end));
  CUDA_CHECK(cudaEventRecord(kernel_start));
  solve(d_input, d_kernel, d_output, input_rows, input_cols, kernel_rows,
        kernel_cols);
  CUDA_CHECK(cudaEventRecord(kernel_end));
  CUDA_CHECK(cudaEventSynchronize(kernel_end));
  if (kernel_time_ms != nullptr) {
    CUDA_CHECK(cudaEventElapsedTime(kernel_time_ms, kernel_start, kernel_end));
  }
  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_end));

  CUDA_CHECK(
      cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_kernel));
  CUDA_CHECK(cudaFree(d_output));
}
