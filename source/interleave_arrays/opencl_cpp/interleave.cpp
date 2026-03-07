#include "interleave.h"
#include "opencl_helpers.h"
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
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
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_A(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * N, const_cast<float*>(h_A));
        cl::Buffer d_B(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * N, const_cast<float*>(h_B));
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * 2 * N);

        cl::Kernel kernel(prog, "interleave_kernel");
        kernel.setArg(0, d_A);
        kernel.setArg(1, d_B);
        kernel.setArg(2, d_output);
        kernel.setArg(3, N);

        size_t local = clppPreferredLocalSize(kernel, dev);
        size_t global = ((size_t)N + local - 1) / local * local;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(global), cl::NDRange(local));
        queue.finish();
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * 2 * N, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
