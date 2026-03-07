#pragma once

void geglu_cpu(const float* input, float* output, int halfN);
void geglu_gpu(const float* h_input, float* h_output, int halfN);
