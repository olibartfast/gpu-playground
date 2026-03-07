#include "softmax.h"
#include "opencl_helpers.h"
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
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
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE);

        cl::Kernel k_find_max(prog, "find_max_partial_kernel");
        size_t local = clppPreferredLocalSize(k_find_max, dev);
        int blocksPerGrid = (N + (int)local - 1) / (int)local;
        size_t sharedSize = local * sizeof(float);

        cl::Buffer d_input(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * N, const_cast<float*>(h_input));
        cl::Buffer d_output(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * N);
        cl::Buffer d_partials(ctx, CL_MEM_READ_WRITE, sizeof(float) * blocksPerGrid);
        cl::Buffer d_max_val(ctx, CL_MEM_READ_WRITE, sizeof(float));
        cl::Buffer d_sum_exp(ctx, CL_MEM_READ_WRITE, sizeof(float));

        size_t globalSize = (size_t)blocksPerGrid * local;
        int isMax = 1, isSum = 0;

        k_find_max.setArg(0, d_input);
        k_find_max.setArg(1, d_partials);
        k_find_max.setArg(2, N);
        k_find_max.setArg(3, cl::Local(sharedSize));
        queue.enqueueNDRangeKernel(k_find_max, cl::NullRange,
                                   cl::NDRange(globalSize), cl::NDRange(local));

        cl::Kernel k_reduce(prog, "reduce_final_kernel");
        k_reduce.setArg(0, d_partials);
        k_reduce.setArg(1, d_max_val);
        k_reduce.setArg(2, blocksPerGrid);
        k_reduce.setArg(3, isMax);
        k_reduce.setArg(4, cl::Local(sharedSize));
        queue.enqueueNDRangeKernel(k_reduce, cl::NullRange,
                                   cl::NDRange(local), cl::NDRange(local));

        cl::Kernel k_sum_exp(prog, "sum_exp_partial_kernel");
        k_sum_exp.setArg(0, d_input);
        k_sum_exp.setArg(1, d_partials);
        k_sum_exp.setArg(2, d_max_val);
        k_sum_exp.setArg(3, N);
        k_sum_exp.setArg(4, cl::Local(sharedSize));
        queue.enqueueNDRangeKernel(k_sum_exp, cl::NullRange,
                                   cl::NDRange(globalSize), cl::NDRange(local));

        k_reduce.setArg(0, d_partials);
        k_reduce.setArg(1, d_sum_exp);
        k_reduce.setArg(2, blocksPerGrid);
        k_reduce.setArg(3, isSum);
        k_reduce.setArg(4, cl::Local(sharedSize));
        queue.enqueueNDRangeKernel(k_reduce, cl::NullRange,
                                   cl::NDRange(local), cl::NDRange(local));

        size_t globalElem = ((size_t)N + local - 1) / local * local;
        cl::Kernel k_elem(prog, "softmax_elementwise_kernel");
        k_elem.setArg(0, d_input);
        k_elem.setArg(1, d_output);
        k_elem.setArg(2, d_max_val);
        k_elem.setArg(3, d_sum_exp);
        k_elem.setArg(4, N);
        queue.enqueueNDRangeKernel(k_elem, cl::NullRange,
                                   cl::NDRange(globalElem), cl::NDRange(local));

        queue.finish();
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * N, h_output);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
