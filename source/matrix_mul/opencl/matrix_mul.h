#pragma once

void matrixMulCPU(float* A, float* B, float* C, int n);
void matrixMulGPU(const float* h_A, const float* h_B, float* h_C, int n);
