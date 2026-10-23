#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define HIP_CHECK(call)                                                     \
  do {                                                                       \
    hipError_t err__ = (call);                                              \
    if (err__ != hipSuccess) {                                              \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,         \
              hipGetErrorString(err__));                                     \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

inline double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
