#pragma once

void clip_cpu(const float* input, float* output, int N, float lo, float hi);
void clip_gpu(const float* h_input, float* h_output, int N, float lo, float hi);
