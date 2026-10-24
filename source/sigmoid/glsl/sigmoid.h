#pragma once
// Host-pointer interface; device resources belong to the backend
// implementation.
void sigmoid_gpu(const float *input, float *output, int n, float *kernel_ms);
void sigmoid2_gpu(const float *input, float *output, int n, float *kernel_ms);
