#include "matrix_mul.h"
#include "opencl_helpers.h"
#include <cstdio>
#include <cstdlib>

#define CL_TILE_SIZE 2

static const std::string KERNEL_SOURCE = R"(
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
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE, "-D TILE_SIZE=2");

        int size = n * n;
        cl::Buffer d_A(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * size, const_cast<float*>(h_A));
        cl::Buffer d_B(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * size, const_cast<float*>(h_B));
        cl::Buffer d_C(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * size);

        cl::Kernel kernel(prog, "matrixMulTiled");
        kernel.setArg(0, d_A);
        kernel.setArg(1, d_B);
        kernel.setArg(2, d_C);
        kernel.setArg(3, n);

        cl::NDRange local(CL_TILE_SIZE, CL_TILE_SIZE);
        cl::NDRange global(((size_t)n + CL_TILE_SIZE - 1) / CL_TILE_SIZE * CL_TILE_SIZE,
                           ((size_t)n + CL_TILE_SIZE - 1) / CL_TILE_SIZE * CL_TILE_SIZE);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        queue.finish();
        queue.enqueueReadBuffer(d_C, CL_TRUE, 0, sizeof(float) * size, h_C);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
