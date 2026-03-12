#include "convolution2d.h"
#include "cuda_helpers.h"

static const int CONV2D_TILE_WIDTH = 16;

static __global__ void conv2d_tiled(const float* input, const float* kernel, float* output,
                             int input_rows, int input_cols,
                             int kernel_rows, int kernel_cols,
                             int out_rows, int out_cols) {
    int out_row = blockIdx.y * CONV2D_TILE_WIDTH + threadIdx.y;
    int out_col = blockIdx.x * CONV2D_TILE_WIDTH + threadIdx.x;

    extern __shared__ float s_input[];
    const int s_width = CONV2D_TILE_WIDTH + kernel_cols - 1;
    const int s_height = CONV2D_TILE_WIDTH + kernel_rows - 1;

    const int tile_origin_row = blockIdx.y * CONV2D_TILE_WIDTH;
    const int tile_origin_col = blockIdx.x * CONV2D_TILE_WIDTH;
    const int num_elements = s_width * s_height;
    const int threads_per_block = CONV2D_TILE_WIDTH * CONV2D_TILE_WIDTH;
    const int tid = threadIdx.y * CONV2D_TILE_WIDTH + threadIdx.x;

    for (int idx = tid; idx < num_elements; idx += threads_per_block) {
        int local_r = idx / s_width;
        int local_c = idx % s_width;
        int global_r = tile_origin_row + local_r;
        int global_c = tile_origin_col + local_c;
        if (global_r < input_rows && global_c < input_cols) {
            s_input[idx] = input[global_r * input_cols + global_c];
        } else {
            s_input[idx] = 0.0f;
        }
    }
    __syncthreads();

    if (out_row < out_rows && out_col < out_cols) {
        float sum = 0.0f;
        for (int kr = 0; kr < kernel_rows; kr++) {
            for (int kc = 0; kc < kernel_cols; kc++) {
                sum += s_input[(threadIdx.y + kr) * s_width + (threadIdx.x + kc)]
                     * kernel[kr * kernel_cols + kc];
            }
        }
        output[out_row * out_cols + out_col] = sum;
    }
}

static void solve(const float* d_input, const float* d_kernel, float* d_output,
           int input_rows, int input_cols,
           int kernel_rows, int kernel_cols) {
    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;
    if (out_rows <= 0 || out_cols <= 0) return;

    dim3 block(CONV2D_TILE_WIDTH, CONV2D_TILE_WIDTH);
    dim3 grid((out_cols + CONV2D_TILE_WIDTH - 1) / CONV2D_TILE_WIDTH,
              (out_rows + CONV2D_TILE_WIDTH - 1) / CONV2D_TILE_WIDTH);

    int s_height = CONV2D_TILE_WIDTH + kernel_rows - 1;
    int s_width = CONV2D_TILE_WIDTH + kernel_cols - 1;
    size_t shared_mem = (size_t)s_height * (size_t)s_width * sizeof(float);

    conv2d_tiled<<<grid, block, shared_mem>>>(
        d_input, d_kernel, d_output,
        input_rows, input_cols,
        kernel_rows, kernel_cols,
        out_rows, out_cols);
    CUDA_CHECK(cudaGetLastError());
}

void convolution2d_cpu(const float* input, const float* kernel, float* output,
                       int input_rows, int input_cols,
                       int kernel_rows, int kernel_cols) {
    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;
    if (out_rows <= 0 || out_cols <= 0) return;

    for (int out_r = 0; out_r < out_rows; out_r++) {
        for (int out_c = 0; out_c < out_cols; out_c++) {
            float sum = 0.0f;
            for (int kr = 0; kr < kernel_rows; kr++) {
                for (int kc = 0; kc < kernel_cols; kc++) {
                    sum += input[(out_r + kr) * input_cols + (out_c + kc)]
                         * kernel[kr * kernel_cols + kc];
                }
            }
            output[out_r * out_cols + out_c] = sum;
        }
    }
}

void convolution2d_gpu(const float* h_input, const float* h_kernel, float* h_output,
                       int input_rows, int input_cols,
                       int kernel_rows, int kernel_cols) {
    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;
    if (out_rows <= 0 || out_cols <= 0) return;

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * input_rows * input_cols));
    CUDA_CHECK(cudaMalloc(&d_kernel, sizeof(float) * kernel_rows * kernel_cols));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * out_rows * out_cols));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(float) * input_rows * input_cols,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kernel_rows * kernel_cols,
                          cudaMemcpyHostToDevice));

    solve(d_input, d_kernel, d_output, input_rows, input_cols, kernel_rows, kernel_cols);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float) * out_rows * out_cols,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));
}
