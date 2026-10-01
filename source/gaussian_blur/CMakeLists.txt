set_source_files_properties(cuda/gaussian_blur.cpp main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(gaussian_blur main.cpp cuda/gaussian_blur.cpp)
target_link_libraries(gaussian_blur PRIVATE cuda_helpers)
