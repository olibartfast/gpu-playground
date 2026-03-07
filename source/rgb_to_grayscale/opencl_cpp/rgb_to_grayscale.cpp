#include "rgb_to_grayscale.h"
#include "opencl_helpers.h"
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
__kernel void rgb_to_grayscale_kernel(__global const float* input,
                                      __global float* output,
                                      int total_pixels) {
    int idx = get_global_id(0);
    int stride = get_global_size(0);
    for (int i = idx; i < total_pixels; i += stride) {
        int input_idx = i * 3;
        float r = input[input_idx];
        float g = input[input_idx + 1];
        float b = input[input_idx + 2];
        output[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}
)";

void rgb_to_grayscale_cpu(const float* input, float* output, int total_pixels) {
    for (int i = 0; i < total_pixels; i++) {
        int idx = i * 3;
        output[i] = 0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
    }
}

void rgb_to_grayscale_gpu(const float* h_input, float* h_output, int total_pixels) {
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * total_pixels * 3, const_cast<float*>(h_input));
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * total_pixels);

        cl::Kernel kernel(prog, "rgb_to_grayscale_kernel");
        kernel.setArg(0, d_input);
        kernel.setArg(1, d_output);
        kernel.setArg(2, total_pixels);

        size_t local = clppPreferredLocalSize(kernel, dev);
        size_t global = ((size_t)total_pixels + local - 1) / local * local;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(global), cl::NDRange(local));
        queue.finish();
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * total_pixels, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
