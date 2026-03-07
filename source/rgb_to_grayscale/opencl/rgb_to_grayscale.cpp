#include "rgb_to_grayscale.h"
#include "opencl_helpers.h"

static const char* KERNEL_SOURCE = R"(
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
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * total_pixels * 3, (void*)h_input, &err);
    CL_CHECK(err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * total_pixels,
                                     nullptr, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "rgb_to_grayscale_kernel", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &total_pixels));

    size_t localSize = 256;
    size_t globalSize = ((size_t)(total_pixels) + 255) / 256 * 256;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * total_pixels,
                                 h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clTeardown(ctx, queue, prog, kernel);
}
