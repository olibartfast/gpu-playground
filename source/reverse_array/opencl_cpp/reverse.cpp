#include "reverse.h"
#include "opencl_helpers.h"
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
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
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, h_input);

        cl::Kernel kernel(prog, "reverse_array_kernel");
        kernel.setArg(0, d_input);
        kernel.setArg(1, N);

        size_t local = clppPreferredLocalSize(kernel, dev);
        size_t global = ((size_t)N + local - 1) / local * local;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(global), cl::NDRange(local));
        queue.finish();
        queue.enqueueReadBuffer(d_input, CL_TRUE, 0, sizeof(float) * N, h_input);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
