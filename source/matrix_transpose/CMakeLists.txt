set_source_files_properties(matrix_transpose.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matrix_transpose main.cpp matrix_transpose.cpp)
target_link_libraries(matrix_transpose PRIVATE cuda_helpers)
