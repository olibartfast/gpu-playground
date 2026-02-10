cmake_minimum_required(VERSION 3.20)

# Enable CUDA for this directory
enable_language(CUDA)

# Set CUDA language property for source files
set_source_files_properties(matrix_mul.cpp PROPERTIES LANGUAGE CUDA)

# Create the matrix_mul executable
add_executable(matrix_mul matrix_mul.cpp)

# Set required CUDA properties and flags
set_target_properties(matrix_mul PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70)  # For Tesla T4 GPU

# Set C++ standard
set_property(TARGET matrix_mul PROPERTY CXX_STANDARD 17)
