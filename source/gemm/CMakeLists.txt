set_source_files_properties(gemm.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(gemm main.cpp gemm.cpp)
target_link_libraries(gemm PRIVATE cuda_helpers)
