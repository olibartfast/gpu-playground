#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err__));                                     \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

inline double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
