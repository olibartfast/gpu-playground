set_source_files_properties(matrix_mul.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matrix_mul main.cpp matrix_mul.cpp)
target_link_libraries(matrix_mul PRIVATE cuda_helpers)
