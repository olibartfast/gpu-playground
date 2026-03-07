#pragma once
#include <cuda_runtime.h>

constexpr unsigned FULL_WARP_MASK = 0xFFFFFFFFu;
constexpr int WARP_SIZE = 32;

__global__ void spmvRowPerThread(const float* __restrict__ A,
                                  const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int M, int N);

__device__ float warpReduceSum(float val);

__global__ void spmvWarpPerRow(const float* __restrict__ A,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                int M, int N);

__global__ void spmvBlockPerRow(const float* __restrict__ A,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int M, int N);

__global__ void spmvHybrid(const float* __restrict__ A,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int M, int N);

void spmvCpu(const float* A, const float* x, float* y, int M, int N);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void spmvGPU(const float* h_A, const float* h_x, float* h_y, int M, int N);
