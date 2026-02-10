cmake_minimum_required(VERSION 3.20)

# Enable CUDA for this directory
enable_language(CUDA)

# Set CUDA language property for source files
set_source_files_properties(geglu.cpp PROPERTIES LANGUAGE CUDA)

# Create the geglu executable
add_executable(geglu geglu.cpp)

# Set required CUDA properties and flags
set_target_properties(geglu PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70)  # For Tesla T4 GPU

# Set C++ standard
set_property(TARGET geglu PROPERTY CXX_STANDARD 17)


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
