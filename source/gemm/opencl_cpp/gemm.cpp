#include "gemm.h"
#include "opencl_helpers.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static const std::string KERNEL_SOURCE = R"(
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
    try {
        cl::Device dev = clppGetGPUDevice();
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);
        cl::Program prog = clppBuildProgram(ctx, dev, KERNEL_SOURCE, "-D TILE_SIZE=16");

        cl::Buffer d_A(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * M * K, const_cast<float*>(h_A));
        cl::Buffer d_B(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * K * N, const_cast<float*>(h_B));
        cl::Buffer d_C(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * M * N, h_C);

        cl::Kernel kernel(prog, "gemmTiled");
        kernel.setArg(0, d_A);
        kernel.setArg(1, d_B);
        kernel.setArg(2, d_C);
        kernel.setArg(3, alpha);
        kernel.setArg(4, beta);
        kernel.setArg(5, M);
        kernel.setArg(6, N);
        kernel.setArg(7, K);

        cl::NDRange local(TILE_SIZE, TILE_SIZE);
        cl::NDRange global(((size_t)N + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
                           ((size_t)M + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        queue.finish();
        queue.enqueueReadBuffer(d_C, CL_TRUE, 0, sizeof(float) * M * N, h_C);
    } catch (const cl::Error& e) {
        fprintf(stderr, "OpenCL error in %s: %s (%d)\n", __func__, e.what(), e.err());
        std::abort();
    }
}
