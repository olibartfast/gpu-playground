#include "gemm.h"
#include "opencl_helpers.h"
#include <cmath>

static const char* KERNEL_SOURCE = R"(
#define TILE_SIZE 16
__kernel void gemmTiled(__global const float* A,
                        __global const float* B,
                        __global float* C,
                        float alpha, float beta,
                        int M, int N, int K) {
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(1) * TILE_SIZE + ty;
    int col = get_group_id(0) * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        int bRow = t * TILE_SIZE + ty;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        float c_val = C[row * N + col];
        C[row * N + col] = alpha * sum + beta * c_val;
    }
}
)";

void gemmCpu(const float* A, const float* B, float* C,
             float alpha, float beta, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

void gemmGPU(const float* h_A, const float* h_B, float* h_C,
             float alpha, float beta, int M, int N, int K) {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE);

    cl_int err;
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * M * K, (void*)h_A, &err);
    CL_CHECK(err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * K * N, (void*)h_B, &err);
    CL_CHECK(err);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * M * N, h_C, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "gemmTiled", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(float), &alpha));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float), &beta));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &M));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &K));

    size_t localSize[2] = {TILE_SIZE, TILE_SIZE};
    size_t globalSize[2] = {((size_t)(N) + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
                             ((size_t)(M) + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE};
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(float) * M * N,
                                 h_C, 0, nullptr, nullptr));

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clTeardown(ctx, queue, prog, kernel);
}
