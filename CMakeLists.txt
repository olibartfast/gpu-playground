cmake_minimum_required(VERSION 3.20)

option(USE_HIP "Build AMD HIP backends instead of CUDA (CMake 3.21+)" OFF)
option(USE_OPENCL     "Build OpenCL C API backends instead of CUDA"        OFF)
option(USE_OPENCL_CPP "Build OpenCL C++ wrapper backends instead of CUDA"  OFF)

if((USE_HIP AND (USE_OPENCL OR USE_OPENCL_CPP)) OR (USE_OPENCL AND USE_OPENCL_CPP))
    message(FATAL_ERROR "Select only one of USE_HIP, USE_OPENCL, USE_OPENCL_CPP")
endif()

if(USE_HIP)
    cmake_minimum_required(VERSION 3.21)
    project(gpu_playground VERSION 0.1.0
        DESCRIPTION "A playground for GPU programming" LANGUAGES CXX HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
elseif(USE_OPENCL OR USE_OPENCL_CPP)
    project(gpu_playground
        VERSION 0.1.0
        DESCRIPTION "A playground for GPU programming"
        LANGUAGES CXX)
    find_package(OpenCL REQUIRED)
else()
    # Default to Turing SM 7.5 (Tesla T4). CUDA 13.x no longer supports
    # offline compilation for architectures older than SM 7.5. Set this before
    # project() so CUDA compiler detection does not fill in an older default.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build for")
    endif()

    project(gpu_playground
        VERSION 0.1.0
        DESCRIPTION "A playground for GPU programming"
        LANGUAGES CXX CUDA)

    # CUDA Architecture Best Practices:
    # - CMake 3.23+: Just don't set it at all, or use "native"
    # - CMake 3.18-3.22: Use "native" or specify a list
    # - For CI/distribution: Set explicit list like 75-real;80-real;86 to avoid PTX overhead
    # - Command line override: cmake -DCMAKE_CUDA_ARCHITECTURES="80;86" ..
    # - The -real suffix compiles actual binary code (SASS) instead of intermediate PTX,
    #   which is faster at runtime but produces larger binaries.
    # See: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT USE_HIP AND NOT USE_OPENCL AND NOT USE_OPENCL_CPP)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Benchmark numbers from an unoptimized build are misleading; default manual
# builds to an optimized build with debug info.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

include(CTest)
include(cmake/GpuKernel.cmake)

# Shared utility library
add_subdirectory(source/utils)

# Per-kernel feature flags and targets
set(GPU_KERNELS
    reverse_array
    matrix_transpose
    softmax
    prefix_sum
    geglu
    gelu
    silu
    swiglu
    gemm
    interleave_arrays
    matrix_mul
    spmv
    value_clipping
    sigmoid
    rgb_to_grayscale
    convolution2d
    deep_learning_inference
    fp16_dot_product
    categorical_cross_entropy
    gaussian_blur
)

foreach(kernel IN LISTS GPU_KERNELS)
    string(TOUPPER "${kernel}" kernel_option)
    option(GPU_ENABLE_${kernel_option} "Build ${kernel} kernel" ON)
    if(GPU_ENABLE_${kernel_option})
        add_subdirectory(source/${kernel})
    endif()
endforeach()

# Python DSL backends are standalone CUDA scripts. They skip (exit 77) when an
# optional dependency is missing, which CTest records as SKIP.
if(BUILD_TESTING AND NOT USE_HIP AND NOT USE_OPENCL AND NOT USE_OPENCL_CPP)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        function(gpu_add_python_test NAME SCRIPT)
            add_test(NAME ${NAME}
                COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/${SCRIPT}")
            set_tests_properties(${NAME} PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                SKIP_RETURN_CODE 77
                LABELS "python;CUDA"
                TIMEOUT 600
            )
        endfunction()

        gpu_add_python_test(sigmoid_triton
            source/sigmoid/python/triton/sigmoid.py)
        gpu_add_python_test(vector_addition_triton
            source/vector_addition/python/triton/vector_addition.py)
        gpu_add_python_test(vector_addition_cute_dsl
            source/vector_addition/python/cute-dsl/vector_addition.py)
    endif()
endif()
