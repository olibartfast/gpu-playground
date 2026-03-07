#pragma once
#include <cuda_runtime.h>

#define TILE_SIZE 16

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define THREAD_TILE 2

__global__ void gemmTiled(const float* A, const float* B, float* C,
                           float alpha, float beta,
                           int M, int N, int K);

__global__ void gemmTiled2x2(const float* A, const float* B, float* C,
                              float alpha, float beta,
                              int M, int N, int K);

__global__ void gemmOptimized(const float* A, const float* B, float* C,
                               float alpha, float beta,
                               int M, int N, int K);

void gemmCpu(const float* A, const float* B, float* C,
             float alpha, float beta,
             int M, int N, int K);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void gemmGPU(const float* h_A, const float* h_B, float* h_C,
             float alpha, float beta,
             int M, int N, int K);
