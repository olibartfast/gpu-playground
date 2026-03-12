#include "convolution2d.h"
#include "opencl_helpers.h"
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
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

    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * input_rows * input_cols,
                           const_cast<float*>(h_input));
        cl::Buffer d_kernel(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * kernel_rows * kernel_cols,
                            const_cast<float*>(h_kernel));
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY,
                            sizeof(float) * out_rows * out_cols);

        cl::Kernel kernel_obj(prog, "convolution2d_kernel");
        kernel_obj.setArg(0, d_input);
        kernel_obj.setArg(1, d_kernel);
        kernel_obj.setArg(2, d_output);
        kernel_obj.setArg(3, input_rows);
        kernel_obj.setArg(4, input_cols);
        kernel_obj.setArg(5, kernel_rows);
        kernel_obj.setArg(6, kernel_cols);
        kernel_obj.setArg(7, out_rows);
        kernel_obj.setArg(8, out_cols);

        cl::NDRange local(16, 16);
        cl::NDRange global(((size_t)out_cols + 15) / 16 * 16,
                           ((size_t)out_rows + 15) / 16 * 16);
        queue.enqueueNDRangeKernel(kernel_obj, cl::NullRange, global, local);
        queue.finish();
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0,
                                sizeof(float) * out_rows * out_cols, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
