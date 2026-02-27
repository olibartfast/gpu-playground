set_source_files_properties(rgb_to_grayscale.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(rgb_to_grayscale main.cpp rgb_to_grayscale.cpp)
target_link_libraries(rgb_to_grayscale PRIVATE cuda_helpers)
