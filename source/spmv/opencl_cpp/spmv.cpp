#include "spmv.h"
#include "opencl_helpers.h"
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
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
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_A(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * M * N, const_cast<float*>(h_A));
        cl::Buffer d_x(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * N, const_cast<float*>(h_x));
        cl::Buffer d_y(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * M);

        cl::Kernel kernel(prog, "spmvRowPerThread");
        kernel.setArg(0, d_A);
        kernel.setArg(1, d_x);
        kernel.setArg(2, d_y);
        kernel.setArg(3, M);
        kernel.setArg(4, N);

        size_t local = clppPreferredLocalSize(kernel, dev);
        size_t global = ((size_t)M + local - 1) / local * local;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(global), cl::NDRange(local));
        queue.finish();
        queue.enqueueReadBuffer(d_y, CL_TRUE, 0, sizeof(float) * M, h_y);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
