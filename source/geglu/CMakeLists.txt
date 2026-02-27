set_source_files_properties(geglu.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(geglu main.cpp geglu.cpp)
target_link_libraries(geglu PRIVATE cuda_helpers)
