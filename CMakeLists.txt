cmake_minimum_required(VERSION 3.20)

option(USE_OPENCL "Build OpenCL backends instead of CUDA" OFF)

if(USE_OPENCL)
    project(gpu_playground
        VERSION 0.1.0
        DESCRIPTION "A playground for GPU programming"
        LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    find_package(OpenCL REQUIRED)
else()
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

    # CUDA Architecture Best Practices:
    # - CMake 3.23+: Just don't set it at all, or use "native"
    # - CMake 3.18-3.22: Use "native" or specify a list
    # - For CI/distribution: Set explicit list like 70-real;75-real;80-real;86 to avoid PTX overhead
    # - Command line override: cmake -DCMAKE_CUDA_ARCHITECTURES="80;86" ..
    # - The -real suffix compiles actual binary code (SASS) instead of intermediate PTX,
    #   which is faster at runtime but produces larger binaries.
    # See: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html

    # Enable separable compilation globally for CUDA
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# Shared utility library
add_subdirectory(source/utils)

# Per-kernel feature flags
option(GPU_ENABLE_REVERSE_ARRAY       "Build reverse_array kernel"       ON)
option(GPU_ENABLE_MATRIX_TRANSPOSE    "Build matrix_transpose kernel"    ON)
option(GPU_ENABLE_SOFTMAX             "Build softmax kernel"             ON)
option(GPU_ENABLE_PREFIX_SUM          "Build prefix_sum kernel"          ON)
option(GPU_ENABLE_GEGLU               "Build geglu kernel"               ON)
option(GPU_ENABLE_SILU                "Build silu kernel"                ON)
option(GPU_ENABLE_SWIGLU              "Build swiglu kernel"              ON)
option(GPU_ENABLE_GEMM                "Build gemm kernel"                ON)
option(GPU_ENABLE_INTERLEAVE_ARRAYS   "Build interleave_arrays kernel"   ON)
option(GPU_ENABLE_MATRIX_MUL          "Build matrix_mul kernel"          ON)
option(GPU_ENABLE_SPMV                "Build spmv kernel"                ON)
option(GPU_ENABLE_VALUE_CLIPPING      "Build value_clipping kernel"      ON)
option(GPU_ENABLE_SIGMOID             "Build sigmoid kernel"             ON)
option(GPU_ENABLE_RGB_TO_GRAYSCALE    "Build rgb_to_grayscale kernel"    ON)

if(GPU_ENABLE_REVERSE_ARRAY)
    add_subdirectory(source/reverse_array)
endif()
if(GPU_ENABLE_MATRIX_TRANSPOSE)
    add_subdirectory(source/matrix_transpose)
endif()
if(GPU_ENABLE_SOFTMAX)
    add_subdirectory(source/softmax)
endif()
if(GPU_ENABLE_PREFIX_SUM)
    add_subdirectory(source/prefix_sum)
endif()
if(GPU_ENABLE_GEGLU)
    add_subdirectory(source/geglu)
endif()
if(GPU_ENABLE_SILU)
    add_subdirectory(source/silu)
endif()
if(GPU_ENABLE_SWIGLU)
    add_subdirectory(source/swiglu)
endif()
if(GPU_ENABLE_GEMM)
    add_subdirectory(source/gemm)
endif()
if(GPU_ENABLE_INTERLEAVE_ARRAYS)
    add_subdirectory(source/interleave_arrays)
endif()
if(GPU_ENABLE_MATRIX_MUL)
    add_subdirectory(source/matrix_mul)
endif()
if(GPU_ENABLE_SPMV)
    add_subdirectory(source/spmv)
endif()
if(GPU_ENABLE_VALUE_CLIPPING)
    add_subdirectory(source/value_clipping)
endif()
if(GPU_ENABLE_SIGMOID)
    add_subdirectory(source/sigmoid)
endif()
if(GPU_ENABLE_RGB_TO_GRAYSCALE)
    add_subdirectory(source/rgb_to_grayscale)
endif()
