set_source_files_properties(swiglu.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(swiglu main.cpp swiglu.cpp)
target_link_libraries(swiglu PRIVATE cuda_helpers)
