#include "geglu.h"
#include "opencl_helpers.h"
#include <cmath>

static const char* KERNEL_SOURCE = R"(
__kernel void geglu_kernel(__global const float* input, __global float* output, int halfN) {
    int i = get_global_id(0);
    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[i + halfN];
        float gelu_x2 = 0.5f * x2 * (1.0f + erf(x2 * 0.70710678f));
        output[i] = x1 * gelu_x2;
    }
}
)";

void geglu_cpu(const float* input, float* output, int halfN) {
    for (int i = 0; i < halfN; i++) {
        float x1 = input[i];
        float x2 = input[i + halfN];
        float gelu_x2 = 0.5f * x2 * (1.0f + erff(x2 * 0.70710678f));
        output[i] = x1 * gelu_x2;
    }
}

void geglu_gpu(const float* h_input, float* h_output, int halfN) {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * 2 * halfN, (void*)h_input, &err);
    CL_CHECK(err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * halfN, nullptr, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "geglu_kernel", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &halfN));

    size_t localSize = 256;
    size_t globalSize = ((size_t)(halfN) + 255) / 256 * 256;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * halfN,
                                 h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clTeardown(ctx, queue, prog, kernel);
}
