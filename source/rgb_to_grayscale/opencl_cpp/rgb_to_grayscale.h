#pragma once

void rgb_to_grayscale_cpu(const float* input, float* output, int total_pixels);
void rgb_to_grayscale_gpu(const float* h_input, float* h_output, int total_pixels);
