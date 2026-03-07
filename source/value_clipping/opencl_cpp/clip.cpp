#include "clip.h"
#include "opencl_helpers.h"
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
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
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, const_cast<float*>(h_input));
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * N);

        cl::Kernel kernel(prog, "clip_kernel");
        kernel.setArg(0, d_input);
        kernel.setArg(1, d_output);
        kernel.setArg(2, N);
        kernel.setArg(3, lo);
        kernel.setArg(4, hi);

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
