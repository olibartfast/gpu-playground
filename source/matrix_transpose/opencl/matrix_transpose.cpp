#include "matrix_transpose.h"
#include "opencl_c_helpers.h"

static const char* KERNEL_SOURCE = R"(
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
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * rows * cols, (void*)h_input, &err);
    CL_CHECK(err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * rows * cols,
                                     nullptr, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "matrix_transpose_kernel", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &rows));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &cols));

    size_t localSize[2] = {16, 16};
    size_t globalSize[2] = {((size_t)(cols) + 15) / 16 * 16,
                             ((size_t)(rows) + 15) / 16 * 16};
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * rows * cols,
                                 h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clTeardown(ctx, queue, prog, kernel);
}
