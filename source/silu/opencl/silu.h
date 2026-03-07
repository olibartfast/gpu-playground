#pragma once

void silu_cpu(const float* input, float* output, int N);
void silu_gpu(const float* h_input, float* h_output, int N);
