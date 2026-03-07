#include "matrix_transpose.h"
#include "opencl_helpers.h"
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
__kernel void matrix_transpose_kernel(__global const float* input,
                                      __global float* output,
                                      int rows, int cols) {
    int c = get_global_id(0);
    int r = get_global_id(1);
    if (r < rows && c < cols)
        output[c * rows + r] = input[r * cols + c];
}
)";

void matrix_transpose_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            output[c * rows + r] = input[r * cols + c];
}

void matrix_transpose_gpu(const float* h_input, float* h_output, int rows, int cols) {
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * rows * cols, const_cast<float*>(h_input));
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * rows * cols);

        cl::Kernel kernel(prog, "matrix_transpose_kernel");
        kernel.setArg(0, d_input);
        kernel.setArg(1, d_output);
        kernel.setArg(2, rows);
        kernel.setArg(3, cols);

        cl::NDRange local(16, 16);
        cl::NDRange global(((size_t)cols + 15) / 16 * 16,
                           ((size_t)rows + 15) / 16 * 16);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        queue.finish();
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * rows * cols, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
