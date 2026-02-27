set_source_files_properties(reverse.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(reverse_array main.cpp reverse.cpp)
target_link_libraries(reverse_array PRIVATE cuda_helpers)
