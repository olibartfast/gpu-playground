#include "reverse.h"
#include "opencl_helpers.h"

static const char* KERNEL_SOURCE = R"(
__kernel void reverse_array_kernel(__global float* input, int N) {
    int tid = get_global_id(0);
    int j = N - tid - 1;
    if (tid < N / 2) {
        float tmp = input[tid];
        input[tid] = input[j];
        input[j] = tmp;
    }
}
)";

void reverse_array_cpu(float* input, int N) {
    int i = 0, j = N - 1;
    while (i < j) {
        float tmp = input[i];
        input[i] = input[j];
        input[j] = tmp;
        i++;
        j--;
    }
}

void reverse_array_gpu(float* h_input, int N) {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * N, h_input, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "reverse_array_kernel", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), &N));

    size_t localSize = clPreferredLocalSize(kernel, dev);
    size_t globalSize = ((size_t)(N) + localSize - 1) / localSize * localSize;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_input, CL_TRUE, 0, sizeof(float) * N,
                                 h_input, 0, nullptr, nullptr));

    clReleaseMemObject(d_input);
    clTeardown(ctx, queue, prog, kernel);
}
