cmake_minimum_required(VERSION 3.20)

# Enable CUDA for this directory
enable_language(CUDA)

# Set CUDA language property for source files
set_source_files_properties(gemm.cpp PROPERTIES LANGUAGE CUDA)

# Create the gemm executable
add_executable(gemm gemm.cpp)

# Set required CUDA properties and flags
set_target_properties(gemm PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70)  # For Tesla T4 GPU

# Set C++ standard
set_property(TARGET gemm PROPERTY CXX_STANDARD 17)
