#pragma once
#include <cuda_runtime.h>

#define TILE_SIZE 2

void matrixMulCPU(float *A, float *B, float *C, int n);

__global__ void matrixMulTiled(float *A, float *B, float *C, int n);
