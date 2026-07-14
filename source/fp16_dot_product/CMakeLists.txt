set_source_files_properties(cuda/fp16_dot_product.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(fp16_dot_product main.cpp cuda/fp16_dot_product.cpp)
target_link_libraries(fp16_dot_product PRIVATE cuda_helpers)
