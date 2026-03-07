#include "prefix_sum.h"
#include "opencl_helpers.h"
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
__kernel void block_inclusive_scan(__global const float* input,
                                   __global float* output,
                                   __global float* block_sums,
                                   int n,
                                   int block_size,
                                   __local float* temp) {
    int tid = get_local_id(0);
    int block_offset = get_group_id(0) * block_size;

    if (tid < block_size && block_offset + tid < n)
        temp[tid] = input[block_offset + tid];
    else
        temp[tid] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < n; offset *= 2) {
        float t = 0.0f;
        if (tid >= offset) t = temp[tid - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < n && tid >= offset) temp[tid] += t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0 && block_offset + block_size - 1 < n)
        block_sums[get_group_id(0)] = temp[block_size - 1];

    if (tid < block_size && block_offset + tid < n)
        output[block_offset + tid] = temp[tid];
}

__kernel void scan_block_sums(__global float* block_sums,
                              int num_blocks,
                              __local float* temp) {
    int tid = get_local_id(0);
    temp[tid] = (tid < num_blocks) ? block_sums[tid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < num_blocks; offset *= 2) {
        float t = 0.0f;
        if (tid >= offset) t = temp[tid - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < num_blocks && tid >= offset) temp[tid] += t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < num_blocks) block_sums[tid] = temp[tid];
}

__kernel void add_block_sums(__global float* output,
                             __global const float* block_sums,
                             int n,
                             int block_size) {
    int tid = get_local_id(0);
    int block_offset = get_group_id(0) * block_size;
    if (get_group_id(0) > 0 && tid < block_size && block_offset + tid < n)
        output[block_offset + tid] += block_sums[get_group_id(0) - 1];
}
)";

void prefix_scan_cpu(float* input, float* output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; i++) output[i] = output[i - 1] + input[i];
}

void prefix_scan_gpu(const float* h_input, float* h_output, int N) {
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, const_cast<float*>(h_input));
        cl::Buffer d_output(ctx, CL_MEM_READ_WRITE, sizeof(float) * N);

        cl::Kernel k_scan(prog, "block_inclusive_scan");
        size_t local = clppPreferredLocalSize(k_scan, dev);
        int threadsPerBlock = (int)local;
        int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        size_t sharedSize = local * sizeof(float);
        int blockSizeArg = threadsPerBlock;

        cl::Buffer d_block_sums(ctx, CL_MEM_READ_WRITE, sizeof(float) * numberOfBlocks);

        k_scan.setArg(0, d_input);
        k_scan.setArg(1, d_output);
        k_scan.setArg(2, d_block_sums);
        k_scan.setArg(3, N);
        k_scan.setArg(4, blockSizeArg);
        k_scan.setArg(5, cl::Local(sharedSize));

        size_t globalSize = (size_t)numberOfBlocks * local;
        queue.enqueueNDRangeKernel(k_scan, cl::NullRange,
                                   cl::NDRange(globalSize), cl::NDRange(local));

        if (numberOfBlocks > 1) {
            cl::Kernel k_scan_sums(prog, "scan_block_sums");
            k_scan_sums.setArg(0, d_block_sums);
            k_scan_sums.setArg(1, numberOfBlocks);
            k_scan_sums.setArg(2, cl::Local(sharedSize));
            queue.enqueueNDRangeKernel(k_scan_sums, cl::NullRange,
                                       cl::NDRange(local), cl::NDRange(local));

            cl::Kernel k_add(prog, "add_block_sums");
            k_add.setArg(0, d_output);
            k_add.setArg(1, d_block_sums);
            k_add.setArg(2, N);
            k_add.setArg(3, blockSizeArg);
            queue.enqueueNDRangeKernel(k_add, cl::NullRange,
                                       cl::NDRange(globalSize), cl::NDRange(local));
        }

        queue.finish();
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * N, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
