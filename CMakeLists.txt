cmake_minimum_required(VERSION 3.20)
project(gpu_playground 
    VERSION 0.1.0
    DESCRIPTION "A playground for GPU programming"
    LANGUAGES CXX CUDA)

# Global settings for all CUDA targets
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Tesla T4 architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

# Enable separable compilation globally for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(reverse_array)
add_subdirectory(matrix_transpose)
add_subdirectory(softmax)
add_subdirectory(prefix_sum)
add_subdirectory(geglu)
add_subdirectory(silu)
add_subdirectory(swiglu)
add_subdirectory(gemm)
add_subdirectory(interleave_arrays)
add_subdirectory(matrix_mul)
add_subdirectory(spmv)
add_subdirectory(value_clipping)
