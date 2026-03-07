#include "swiglu.h"
#include "opencl_helpers.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
__kernel void swiglu_kernel(__global const float* input, __global float* output, int N) {
    int i = get_global_id(0);
    if (i < N / 2) {
        float x1 = input[i];
        float x2 = input[i + N / 2];
        float silu = x1 / (1.0f + native_exp(-x1));
        output[i] = silu * x2;
    }
}
)";

void swiglu_cpu(const float* input, float* output, int N) {
    for (int i = 0; i < N / 2; i++) {
        float x1 = input[i];
        float x2 = input[i + N / 2];
        float silu = x1 / (1.0f + expf(-x1));
        output[i] = silu * x2;
    }
}

void swiglu_gpu(const float* h_input, float* h_output, int N) {
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, const_cast<float*>(h_input));
        int halfN = N / 2;
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * halfN);

        cl::Kernel kernel(prog, "swiglu_kernel");
        kernel.setArg(0, d_input);
        kernel.setArg(1, d_output);
        kernel.setArg(2, N);

        size_t local = clppPreferredLocalSize(kernel, dev);
        size_t global = ((size_t)halfN + local - 1) / local * local;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(global), cl::NDRange(local));
        queue.finish();
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * halfN, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
