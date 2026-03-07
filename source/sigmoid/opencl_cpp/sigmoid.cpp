#include "sigmoid.h"
#include "opencl_helpers.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
__kernel void sigmoid_kernel(__global const float* X, __global float* Y, int N) {
    int i = get_global_id(0);
    if (i < N) Y[i] = 1.0f / (1.0f + native_exp(-X[i]));
}

__kernel void sigmoid_kernel2(__global const float* X, __global float* Y, int N) {
    int i = (int)get_global_id(0) * 4;
    if (i + 3 < N) {
        float4 x = vload4(get_global_id(0), X);
        float4 y;
        y.x = 1.0f / (1.0f + native_exp(-x.x));
        y.y = 1.0f / (1.0f + native_exp(-x.y));
        y.z = 1.0f / (1.0f + native_exp(-x.z));
        y.w = 1.0f / (1.0f + native_exp(-x.w));
        vstore4(y, get_global_id(0), Y);
    } else {
        for (int j = i; j < N; j++) {
            Y[j] = 1.0f / (1.0f + native_exp(-X[j]));
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
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, const_cast<float*>(h_input));
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * N);

        cl::Kernel kernel(prog, "sigmoid_kernel");
        kernel.setArg(0, d_input);
        kernel.setArg(1, d_output);
        kernel.setArg(2, N);

        size_t local = clppPreferredLocalSize(kernel, dev);
        size_t global = ((size_t)N + local - 1) / local * local;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(global), cl::NDRange(local));
        queue.finish();
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * N, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}

void sigmoid2_gpu(const float* h_input, float* h_output, int N) {
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, const_cast<float*>(h_input));
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * N);

        cl::Kernel kernel(prog, "sigmoid_kernel2");
        kernel.setArg(0, d_input);
        kernel.setArg(1, d_output);
        kernel.setArg(2, N);

        size_t local = clppPreferredLocalSize(kernel, dev);
        int nVec = N / 4;
        size_t global = nVec > 0 ? ((size_t)nVec + local - 1) / local * local : local;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(global), cl::NDRange(local));
        queue.finish();
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * N, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
