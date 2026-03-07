#pragma once

void spmvCpu(const float* A, const float* x, float* y, int M, int N);
void spmvGPU(const float* h_A, const float* h_x, float* h_y, int M, int N);
