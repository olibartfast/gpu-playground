#include "convolution2d.h"
#include "opencl_c_helpers.h"

static const char* KERNEL_SOURCE = R"(
__kernel void convolution2d_kernel(__global const float* input,
                                   __global const float* kernel,
                                   __global float* output,
                                   int input_rows, int input_cols,
                                   int kernel_rows, int kernel_cols,
                                   int out_rows, int out_cols) {
    int out_col = get_global_id(0);
    int out_row = get_global_id(1);
    if (out_row < out_rows && out_col < out_cols) {
        float sum = 0.0f;
        for (int kr = 0; kr < kernel_rows; kr++) {
            for (int kc = 0; kc < kernel_cols; kc++) {
                sum += input[(out_row + kr) * input_cols + (out_col + kc)]
                     * kernel[kr * kernel_cols + kc];
            }
        }
        output[out_row * out_cols + out_col] = sum;
    }
}
)";

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

    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * input_rows * input_cols,
                                    (void*)h_input, &err);
    CL_CHECK(err);
    cl_mem d_kernel = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * kernel_rows * kernel_cols,
                                     (void*)h_kernel, &err);
    CL_CHECK(err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                     sizeof(float) * out_rows * out_cols, nullptr, &err);
    CL_CHECK(err);

    cl_kernel kernel_obj = clCreateKernel(prog, "convolution2d_kernel", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel_obj, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel_obj, 1, sizeof(cl_mem), &d_kernel));
    CL_CHECK(clSetKernelArg(kernel_obj, 2, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel_obj, 3, sizeof(int), &input_rows));
    CL_CHECK(clSetKernelArg(kernel_obj, 4, sizeof(int), &input_cols));
    CL_CHECK(clSetKernelArg(kernel_obj, 5, sizeof(int), &kernel_rows));
    CL_CHECK(clSetKernelArg(kernel_obj, 6, sizeof(int), &kernel_cols));
    CL_CHECK(clSetKernelArg(kernel_obj, 7, sizeof(int), &out_rows));
    CL_CHECK(clSetKernelArg(kernel_obj, 8, sizeof(int), &out_cols));

    size_t local[2] = {16, 16};
    size_t global[2] = {
        ((size_t)out_cols + local[0] - 1) / local[0] * local[0],
        ((size_t)out_rows + local[1] - 1) / local[1] * local[1]
    };
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel_obj, 2, nullptr, global, local,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                                 sizeof(float) * out_rows * out_cols,
                                 h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_kernel);
    clReleaseMemObject(d_output);
    clTeardown(ctx, queue, prog, kernel_obj);
}
