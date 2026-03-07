#pragma once

void swiglu_cpu(const float* input, float* output, int N);
void swiglu_gpu(const float* h_input, float* h_output, int N);
