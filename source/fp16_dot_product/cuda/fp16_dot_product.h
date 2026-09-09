#pragma once

float fp16_dot_product_cpu(const float* a, const float* b, int n);

// Converts the host inputs to FP16, multiplies and accumulates in FP32, rounds
// the final result to FP16, and returns that result as a host float. When
// kernel_time_ms is non-null, it receives device execution time without host
// conversion, allocation, or transfers.
float fp16_dot_product_gpu(const float* h_a,
                           const float* h_b,
                           int n,
                           float* kernel_time_ms = nullptr);
