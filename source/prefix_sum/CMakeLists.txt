set_source_files_properties(prefix_sum.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(prefix_sum main.cpp prefix_sum.cpp)
target_link_libraries(prefix_sum PRIVATE cuda_helpers)
