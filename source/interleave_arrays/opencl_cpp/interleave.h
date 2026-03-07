#pragma once

void interleave_cpu(const float* A, const float* B, float* output, int N);
void interleave_gpu(const float* h_A, const float* h_B, float* h_output, int N);
