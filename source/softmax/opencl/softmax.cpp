#include "softmax.h"
#include "opencl_c_helpers.h"
#include <cmath>
#include <cfloat>

static const char* KERNEL_SOURCE = R"(
__kernel void find_max_partial_kernel(__global const float* input,
                                      __global float* partial_maxs,
                                      int N,
                                      __local float* sdata) {
    int tid = get_local_id(0);
    int i = get_group_id(0) * get_local_size(0) + tid;
    int gridSize = get_num_groups(0) * get_local_size(0);

    float my_max = -INFINITY;
    while (i < N) {
        my_max = fmax(my_max, input[i]);
        i += gridSize;
    }
    sdata[tid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) partial_maxs[get_group_id(0)] = sdata[0];
}

__kernel void sum_exp_partial_kernel(__global const float* input,
                                     __global float* partial_sums,
                                     __global const float* max_val,
                                     int N,
                                     __local float* sdata) {
    int tid = get_local_id(0);
    int i = get_group_id(0) * get_local_size(0) + tid;
    int gridSize = get_num_groups(0) * get_local_size(0);

    float my_sum = 0.0f;
    while (i < N) {
        my_sum += native_exp(input[i] - (*max_val));
        i += gridSize;
    }
    sdata[tid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) partial_sums[get_group_id(0)] = sdata[0];
}

__kernel void reduce_final_kernel(__global float* partials,
                                  __global float* result,
                                  int N,
                                  int is_max_reduction,
                                  __local float* sdata) {
    int tid = get_local_id(0);
    float my_val = is_max_reduction ? -INFINITY : 0.0f;
    for (int i = tid; i < N; i += get_local_size(0)) {
        if (is_max_reduction) my_val = fmax(my_val, partials[i]);
        else                  my_val += partials[i];
    }
    sdata[tid] = my_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (is_max_reduction) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
            else                  sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) result[0] = sdata[0];
}

__kernel void softmax_elementwise_kernel(__global const float* input,
                                         __global float* output,
                                         __global const float* max_val,
                                         __global const float* sum_exp,
                                         int N) {
    int i = get_global_id(0);
    if (i < N) {
        output[i] = native_exp(input[i] - (*max_val)) / (*sum_exp);
    }
}
)";

void softmax_cpu(const float* input, float* output, int N) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) max_val = fmaxf(max_val, input[i]);
    float sum_exp = 0.0f;
    for (int i = 0; i < N; i++) sum_exp += expf(input[i] - max_val);
    for (int i = 0; i < N; i++) output[i] = expf(input[i] - max_val) / sum_exp;
}

void softmax_gpu(const float* h_input, float* h_output, int N) {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_kernel k_find_max = clCreateKernel(prog, "find_max_partial_kernel", &err); CL_CHECK(err);
    size_t localSize = clPreferredLocalSize(k_find_max, dev);
    int blocksPerGrid = (N + (int)localSize - 1) / (int)localSize;

    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * N, (void*)h_input, &err);
    CL_CHECK(err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * N, nullptr, &err);
    CL_CHECK(err);
    cl_mem d_partials = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       sizeof(float) * blocksPerGrid, nullptr, &err);
    CL_CHECK(err);
    cl_mem d_max_val = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float), nullptr, &err);
    CL_CHECK(err);
    cl_mem d_sum_exp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float), nullptr, &err);
    CL_CHECK(err);

    size_t globalSize = (size_t)blocksPerGrid * localSize;
    size_t sharedSize = localSize * sizeof(float);
    int isMax = 1, isSum = 0;

    CL_CHECK(clSetKernelArg(k_find_max, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(k_find_max, 1, sizeof(cl_mem), &d_partials));
    CL_CHECK(clSetKernelArg(k_find_max, 2, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(k_find_max, 3, sharedSize, nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_find_max, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));

    cl_kernel k_reduce = clCreateKernel(prog, "reduce_final_kernel", &err); CL_CHECK(err);
    CL_CHECK(clSetKernelArg(k_reduce, 0, sizeof(cl_mem), &d_partials));
    CL_CHECK(clSetKernelArg(k_reduce, 1, sizeof(cl_mem), &d_max_val));
    CL_CHECK(clSetKernelArg(k_reduce, 2, sizeof(int), &blocksPerGrid));
    CL_CHECK(clSetKernelArg(k_reduce, 3, sizeof(int), &isMax));
    CL_CHECK(clSetKernelArg(k_reduce, 4, sharedSize, nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_reduce, 1, nullptr, &localSize, &localSize,
                                    0, nullptr, nullptr));

    cl_kernel k_sum_exp = clCreateKernel(prog, "sum_exp_partial_kernel", &err); CL_CHECK(err);
    CL_CHECK(clSetKernelArg(k_sum_exp, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(k_sum_exp, 1, sizeof(cl_mem), &d_partials));
    CL_CHECK(clSetKernelArg(k_sum_exp, 2, sizeof(cl_mem), &d_max_val));
    CL_CHECK(clSetKernelArg(k_sum_exp, 3, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(k_sum_exp, 4, sharedSize, nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_sum_exp, 1, nullptr, &globalSize, &localSize,
                                    0, nullptr, nullptr));

    CL_CHECK(clSetKernelArg(k_reduce, 0, sizeof(cl_mem), &d_partials));
    CL_CHECK(clSetKernelArg(k_reduce, 1, sizeof(cl_mem), &d_sum_exp));
    CL_CHECK(clSetKernelArg(k_reduce, 2, sizeof(int), &blocksPerGrid));
    CL_CHECK(clSetKernelArg(k_reduce, 3, sizeof(int), &isSum));
    CL_CHECK(clSetKernelArg(k_reduce, 4, sharedSize, nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_reduce, 1, nullptr, &localSize, &localSize,
                                    0, nullptr, nullptr));

    size_t globalElem = ((size_t)(N) + localSize - 1) / localSize * localSize;
    cl_kernel k_elem = clCreateKernel(prog, "softmax_elementwise_kernel", &err); CL_CHECK(err);
    CL_CHECK(clSetKernelArg(k_elem, 0, sizeof(cl_mem), &d_input));
    CL_CHECK(clSetKernelArg(k_elem, 1, sizeof(cl_mem), &d_output));
    CL_CHECK(clSetKernelArg(k_elem, 2, sizeof(cl_mem), &d_max_val));
    CL_CHECK(clSetKernelArg(k_elem, 3, sizeof(cl_mem), &d_sum_exp));
    CL_CHECK(clSetKernelArg(k_elem, 4, sizeof(int), &N));
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_elem, 1, nullptr, &globalElem, &localSize,
                                    0, nullptr, nullptr));

    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * N,
                                 h_output, 0, nullptr, nullptr));

    clReleaseKernel(k_find_max);
    clReleaseKernel(k_sum_exp);
    clReleaseKernel(k_elem);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_partials);
    clReleaseMemObject(d_max_val);
    clReleaseMemObject(d_sum_exp);
    clTeardown(ctx, queue, prog, k_reduce);
}
