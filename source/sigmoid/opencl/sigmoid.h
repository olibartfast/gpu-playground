#pragma once

void sigmoid_cpu(const float* input, float* output, int N);
void sigmoid_gpu(const float* h_input, float* h_output, int N);
void sigmoid2_gpu(const float* h_input, float* h_output, int N);
