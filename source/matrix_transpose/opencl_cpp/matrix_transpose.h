#pragma once

void matrix_transpose_cpu(const float* input, float* output, int rows, int cols);
void matrix_transpose_gpu(const float* h_input, float* h_output, int rows, int cols);
