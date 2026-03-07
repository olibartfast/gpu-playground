#include "clip.h"
#include "opencl_c_helpers.h"

static const char* KERNEL_SOURCE = R"(
__kernel void clip_kernel(__global const float* input, __global float* output,
                          int N, float lo, float hi) {
    int idx = get_global_id(0);
    if (idx < N) {
        output[idx] = fmin(fmax(input[idx], lo), hi);
    }
}
)";

void clip_cpu(const float* input, float* output, int N, float lo, float hi) {
    for (int i = 0; i < N; i++) {
        float val = input[i];
        if (val < lo)       output[i] = lo;
        else if (val > hi)  output[i] = hi;
        else                output[i] = val;
    }
}

void clip_gpu(const float* h_input, float* h_output, int N, float lo, float hi) {
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

    cl_kernel kernel = clCreateKernel(prog, "clip_kernel", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(float), &lo));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float), &hi));

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
