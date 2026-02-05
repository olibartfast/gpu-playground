cmake_minimum_required(VERSION 3.20)

enable_language(CUDA)

set_source_files_properties(geglu.cpp PROPERTIES LANGUAGE CUDA)

add_executable(geglu geglu.cpp)

# Let CMake handle architecture detection automatically
# If not set, CMake 3.23+ will detect the GPU on the build machine
# For older CMake or to compile for multiple architectures, uncomment:
# set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Or use special values:
# set(CMAKE_CUDA_ARCHITECTURES native)  # Detect current GPU
# set(CMAKE_CUDA_ARCHITECTURES all)     # All architectures (slow build)
# set(CMAKE_CUDA_ARCHITECTURES all-major)  # Major versions only

set_target_properties(geglu PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17)

# check also
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
# cmake_minimum_required(VERSION 3.23)  # For automatic detection

# enable_language(CUDA)

# set_source_files_properties(geglu.cpp PROPERTIES LANGUAGE CUDA)

# add_executable(geglu geglu.cpp)

# # Use 'native' to auto-detect, or allow override via command line
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#     set(CMAKE_CUDA_ARCHITECTURES native)
# endif()

# set_target_properties(silu PROPERTIES 
#     CUDA_SEPARABLE_COMPILATION ON
#     CXX_STANDARD 17)    


# Best practices:
# CMake 3.23+: Just don't set it at all, or use native
# CMake 3.18-3.22: Use native or specify a list
# For CI/distribution: Set explicit list like 70-real;75-real;80-real;86 to avoid PTX overhead
# Command line override: cmake -DCMAKE_CUDA_ARCHITECTURES="80;86" ..

# The -real suffix compiles actual binary code (SASS) instead of intermediate PTX, which is faster at runtime but produces larger binaries.