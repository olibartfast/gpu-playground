cmake_minimum_required(VERSION 3.20)

# Enable CUDA for this directory
enable_language(CUDA)

# Set CUDA language property for source files
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Create the spmv executable
add_executable(spmv spmv.cpp)

# Set required CUDA properties and flags
set_target_properties(spmv PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70)  # For Tesla T4 GPU

# Set C++ standard
set_property(TARGET spmv PROPERTY CXX_STANDARD 17)
