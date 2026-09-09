#pragma once

float categorical_cross_entropy_cpu(const float *logits, const int *true_labels,
                                    int n, int c);

// Copies host inputs to the device and returns the scalar loss on the host.
// kernel_time_ms excludes allocation and transfers when non-null.
float categorical_cross_entropy_gpu(const float *logits, const int *true_labels,
                                    int n, int c,
                                    float *kernel_time_ms = nullptr);

// LeetGPU-compatible entrypoint. All pointers are device pointers.
extern "C" void solve(const float *logits, const int *true_labels, float *loss,
                      int n, int c);
