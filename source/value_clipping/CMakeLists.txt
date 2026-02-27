set_source_files_properties(clip.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(value_clipping main.cpp clip.cpp)
target_link_libraries(value_clipping PRIVATE cuda_helpers)
