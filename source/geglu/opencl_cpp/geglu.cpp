#include "geglu.h"
#include "opencl_helpers.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
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
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * 2 * halfN, const_cast<float*>(h_input));
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * halfN);

        cl::Kernel kernel(prog, "geglu_kernel");
        kernel.setArg(0, d_input);
        kernel.setArg(1, d_output);
        kernel.setArg(2, halfN);

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
