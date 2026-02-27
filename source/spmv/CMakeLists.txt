set_source_files_properties(spmv.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv main.cpp spmv.cpp)
target_link_libraries(spmv PRIVATE cuda_helpers)
