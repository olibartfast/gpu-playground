#include "interleave.h"
#include "opencl_c_helpers.h"

static const char* KERNEL_SOURCE = R"(
__kernel void interleave_kernel(__global const float* A, __global const float* B,
                                __global float* output, int N) {
    int idx = get_global_id(0);
    if (idx < N) {
        output[2 * idx]     = A[idx];
        output[2 * idx + 1] = B[idx];
    }
}
)";

void interleave_cpu(const float* A, const float* B, float* output, int N) {
    for (int i = 0; i < N; i++) {
        output[2 * i]     = A[i];
        output[2 * i + 1] = B[i];
    }
}

void interleave_gpu(const float* h_A, const float* h_B, float* h_output, int N) {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * N, (void*)h_A, &err);
    CL_CHECK(err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * N, (void*)h_B, &err);
    CL_CHECK(err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * 2 * N,
                                     nullptr, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "interleave_kernel", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &N));

    size_t localSize = clPreferredLocalSize(kernel, dev);
    size_t globalSize = ((size_t)(N) + localSize - 1) / localSize * localSize;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * 2 * N,
                                 h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_output);
    clTeardown(ctx, queue, prog, kernel);
}
