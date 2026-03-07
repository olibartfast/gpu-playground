#include "silu.h"
#include "opencl_c_helpers.h"
#include <cmath>

static const char* KERNEL_SOURCE = R"(
__kernel void silu_kernel(__global const float* input, __global float* output, int N) {
    int i = get_global_id(0);
    if (i < N) {
        float sigma = 1.0f / (1.0f + native_exp(-input[i]));
        output[i] = input[i] * sigma;
    }
}
)";

void silu_cpu(const float* input, float* output, int N) {
    for (int i = 0; i < N; i++) {
        float sigma = 1.0f / (1.0f + expf(-input[i]));
        output[i] = input[i] * sigma;
    }
}

void silu_gpu(const float* h_input, float* h_output, int N) {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * N, (void*)h_input, &err);
    CL_CHECK(err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * N, nullptr, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "silu_kernel", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &N));

    size_t localSize = clPreferredLocalSize(kernel, dev);
    size_t globalSize = ((size_t)(N) + localSize - 1) / localSize * localSize;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * N,
                                 h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clTeardown(ctx, queue, prog, kernel);
}
