#pragma once
#include <cuda_runtime.h>

#define TILE_SIZE 2

void matrixMulCPU(float *A, float *B, float *C, int n);

__global__ void matrixMulTiled(float *A, float *B, float *C, int n);

// Host-pointer wrapper (backend-agnostic API used by main.cpp)
void matrixMulGPU(const float* h_A, const float* h_B, float* h_C, int n);
