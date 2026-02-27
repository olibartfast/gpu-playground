set_source_files_properties(interleave.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(interleave_arrays main.cpp interleave.cpp)
target_link_libraries(interleave_arrays PRIVATE cuda_helpers)
