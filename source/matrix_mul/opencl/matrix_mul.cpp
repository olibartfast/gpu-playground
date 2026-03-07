#include "matrix_mul.h"
#include "opencl_helpers.h"

#define CL_TILE_SIZE 2

static const char* KERNEL_SOURCE = R"(
__kernel void matrixMulTiled(__global float* A, __global float* B, __global float* C, int n) {
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    int row = get_group_id(1) * TILE_SIZE + get_local_id(1);
    int col = get_group_id(0) * TILE_SIZE + get_local_id(0);

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < n && (t * TILE_SIZE + get_local_id(0)) < n)
            tileA[get_local_id(1)][get_local_id(0)] = A[row * n + t * TILE_SIZE + get_local_id(0)];
        else
            tileA[get_local_id(1)][get_local_id(0)] = 0.0f;

        if (col < n && (t * TILE_SIZE + get_local_id(1)) < n)
            tileB[get_local_id(1)][get_local_id(0)] = B[(t * TILE_SIZE + get_local_id(1)) * n + col];
        else
            tileB[get_local_id(1)][get_local_id(0)] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < n && col < n) {
            for (int k = 0; k < TILE_SIZE; k++)
                sum += tileA[get_local_id(1)][k] * tileB[k][get_local_id(0)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}
)";

void matrixMulCPU(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

void matrixMulGPU(const float* h_A, const float* h_B, float* h_C, int n) {
    cl_context ctx;
    cl_command_queue queue;
    cl_device_id dev = clSetupGPU(ctx, queue);
    cl_program prog = clBuildFromSource(ctx, dev, KERNEL_SOURCE, "-D TILE_SIZE=2");

    int size = n * n;
    cl_int err;
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * size, (void*)h_A, &err);
    CL_CHECK(err);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * size, (void*)h_B, &err);
    CL_CHECK(err);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * size, nullptr, &err);
    CL_CHECK(err);

    cl_kernel kernel = clCreateKernel(prog, "matrixMulTiled", &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n));

    size_t localSize[2] = {CL_TILE_SIZE, CL_TILE_SIZE};
    size_t globalSize[2] = {((size_t)(n) + CL_TILE_SIZE - 1) / CL_TILE_SIZE * CL_TILE_SIZE,
                             ((size_t)(n) + CL_TILE_SIZE - 1) / CL_TILE_SIZE * CL_TILE_SIZE};
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize,
                                    0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    CL_CHECK(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(float) * size,
                                 h_C, 0, nullptr, nullptr));

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clTeardown(ctx, queue, prog, kernel);
}
