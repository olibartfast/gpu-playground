set_source_files_properties(sigmoid.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(sigmoid main.cpp sigmoid.cpp)
target_link_libraries(sigmoid PRIVATE cuda_helpers)
