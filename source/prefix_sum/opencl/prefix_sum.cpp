#include "prefix_sum.h"
#include "opencl_helpers.h"

static const char* KERNEL_SOURCE = R"(
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
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * N, (void*)h_input, &err);
    CL_CHECK(err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * N, nullptr, &err);
    CL_CHECK(err);

    // Create scan kernel early to query preferred work-group size for buffer sizing
    cl_kernel k_scan = clCreateKernel(prog, "block_inclusive_scan", &err); CL_CHECK(err);
    size_t localSize = clPreferredLocalSize(k_scan, dev);
    int threadsPerBlock = (int)localSize;
    int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    cl_mem d_block_sums = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                         sizeof(float) * numberOfBlocks, nullptr, &err);
    CL_CHECK(err);

    size_t sharedSize = localSize * sizeof(float);
    int blockSizeArg = threadsPerBlock;

    CL_CHECK(clSetKernelArg(k_scan, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(k_scan, 1, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(k_scan, 2, sizeof(cl_mem), &d_block_sums));
    CL_CHECK(clSetKernelArg(k_scan, 3, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(k_scan, 4, sizeof(int), &blockSizeArg));
    CL_CHECK(clSetKernelArg(k_scan, 5, sharedSize, nullptr));

    size_t globalSize = (size_t)numberOfBlocks * localSize;
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_scan, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));

    if (numberOfBlocks > 1) {
        size_t scanBlocksGlobal = localSize; // single block for block_sums
        cl_kernel k_scan_sums = clCreateKernel(prog, "scan_block_sums", &err); CL_CHECK(err);
        CL_CHECK(clSetKernelArg(k_scan_sums, 0, sizeof(cl_mem), &d_block_sums));
        CL_CHECK(clSetKernelArg(k_scan_sums, 1, sizeof(int), &numberOfBlocks));
        CL_CHECK(clSetKernelArg(k_scan_sums, 2, sharedSize, nullptr));
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_scan_sums, 1, nullptr, &scanBlocksGlobal,
                                        &localSize, 0, nullptr, nullptr));

        cl_kernel k_add = clCreateKernel(prog, "add_block_sums", &err); CL_CHECK(err);
        CL_CHECK(clSetKernelArg(k_add, 0, sizeof(cl_mem), &d_output));
        CL_CHECK(clSetKernelArg(k_add, 1, sizeof(cl_mem), &d_block_sums));
        CL_CHECK(clSetKernelArg(k_add, 2, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(k_add, 3, sizeof(int), &blockSizeArg));
        CL_CHECK(clEnqueueNDRangeKernel(queue, k_add, 1, nullptr, &globalSize, &localSize,
                                        0, nullptr, nullptr));

        CL_CHECK(clFinish(queue));
        clReleaseKernel(k_scan_sums);
        clReleaseKernel(k_add);
    } else {
        CL_CHECK(clFinish(queue));
    }

    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * N,
                                 h_output, 0, nullptr, nullptr));

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_block_sums);
    clTeardown(ctx, queue, prog, k_scan);
}
