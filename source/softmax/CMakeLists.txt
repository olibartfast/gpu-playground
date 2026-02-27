set_source_files_properties(softmax.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(softmax main.cpp softmax.cpp)
target_link_libraries(softmax PRIVATE cuda_helpers)
