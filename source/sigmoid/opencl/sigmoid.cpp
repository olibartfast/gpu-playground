#include "sigmoid.h"
#include "opencl_helpers.h"
#include <cmath>

static const char* KERNEL_SOURCE = R"(
__kernel void sigmoid_kernel(__global const float* X, __global float* Y, int N) {
    int i = get_global_id(0);
    if (i < N) Y[i] = 1.0f / (1.0f + exp(-X[i]));
}

__kernel void sigmoid_kernel2(__global const float* X, __global float* Y, int N) {
    int i = (int)get_global_id(0) * 4;
    if (i + 3 < N) {
        float4 x = vload4(get_global_id(0), X);
        float4 y;
        y.x = 1.0f / (1.0f + exp(-x.x));
        y.y = 1.0f / (1.0f + exp(-x.y));
        y.z = 1.0f / (1.0f + exp(-x.z));
        y.w = 1.0f / (1.0f + exp(-x.w));
        vstore4(y, get_global_id(0), Y);
    } else {
        for (int j = i; j < N; j++) {
            Y[j] = 1.0f / (1.0f + exp(-X[j]));
        }
    }
}
)";

void sigmoid_cpu(const float* input, float* output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

void sigmoid_gpu(const float* h_input, float* h_output, int N) {
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

    cl_kernel kernel = clCreateKernel(prog, "sigmoid_kernel", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &N));

    size_t localSize = 256;
    size_t globalSize = ((size_t)(N) + 255) / 256 * 256;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * N,
                                 h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clTeardown(ctx, queue, prog, kernel);
}

void sigmoid2_gpu(const float* h_input, float* h_output, int N) {
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

    cl_kernel kernel = clCreateKernel(prog, "sigmoid_kernel2", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &N));

    size_t localSize = 256;
    int nVec = N / 4;
    size_t globalSize = nVec > 0 ? ((size_t)(nVec) + 255) / 256 * 256 : localSize;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * N,
                                 h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clTeardown(ctx, queue, prog, kernel);
}
