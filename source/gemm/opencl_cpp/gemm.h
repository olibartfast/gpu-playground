#pragma once

#define TILE_SIZE 16

void gemmCpu(const float* A, const float* B, float* C,
             float alpha, float beta,
             int M, int N, int K);

void gemmGPU(const float* h_A, const float* h_B, float* h_C,
             float alpha, float beta,
             int M, int N, int K);
