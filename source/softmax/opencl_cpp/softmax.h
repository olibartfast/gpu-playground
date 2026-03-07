#pragma once

void softmax_cpu(const float* input, float* output, int N);
void softmax_gpu(const float* h_input, float* h_output, int N);
