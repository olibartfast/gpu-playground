set_source_files_properties(silu.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(silu main.cpp silu.cpp)
target_link_libraries(silu PRIVATE cuda_helpers)
