#pragma once

void prefix_scan_cpu(float* input, float* output, int N);
void prefix_scan_gpu(const float* h_input, float* h_output, int N);
