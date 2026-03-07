#include "spmv.h"
#include "opencl_helpers.h"

static const char* KERNEL_SOURCE = R"(
__kernel void spmvRowPerThread(__global const float* A,
                               __global const float* x,
                               __global float* y,
                               int M, int N) {
    int row = get_global_id(0);
    if (row < M) {
        float sum = 0.0f;
        int rowStart = row * N;
        for (int col = 0; col < N; col++) {
            float val = A[rowStart + col];
            if (val != 0.0f) sum += val * x[col];
        }
        y[row] = sum;
    }
}
)";

void spmvCpu(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += A[i * N + j] * x[j];
        y[i] = sum;
    }
}

void spmvGPU(const float* h_A, const float* h_x, float* h_y, int M, int N) {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * M * N, (void*)h_A, &err);
    CL_CHECK(err);
    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * N, (void*)h_x, &err);
    CL_CHECK(err);
    cl_mem d_y = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * M, nullptr, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "spmvRowPerThread", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_x));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_y));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &M));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &N));

    size_t localSize = clPreferredLocalSize(kernel, dev);
    size_t globalSize = ((size_t)(M) + localSize - 1) / localSize * localSize;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, sizeof(float) * M,
                                 h_y, 0, nullptr, nullptr));

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clTeardown(ctx, queue, prog, kernel);
}
