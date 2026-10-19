include_guard(GLOBAL)

# gpu_add_kernel(<name>
#   [SOURCE_NAME <stem>]
#   [BACKENDS <CUDA|OPENCL|OPENCL_CPP>...]
#   [EXTRA_SOURCES <source>...]
#   [COMPILE_DEFINITIONS <definition>...]
#   [COMPILE_OPTIONS <option>...]
#   [NO_TEST]
# )
#
# Adds one kernel target for the backend selected by USE_OPENCL_CPP / USE_OPENCL.
# The backend source defaults to <backend>/<name>.cpp; SOURCE_NAME overrides the
# stem for kernels whose files use a shorter name. A backend that is not
# implemented is skipped with a status message instead of breaking the build.
#
# Every target is also registered with CTest unless NO_TEST is given. The
# binary's exit code is the test result, matching the harness contract.
function(gpu_add_kernel NAME)
    cmake_parse_arguments(
        ARG
        "NO_TEST"
        "SOURCE_NAME"
        "BACKENDS;EXTRA_SOURCES;COMPILE_DEFINITIONS;COMPILE_OPTIONS"
        ${ARGN}
    )

    if(ARG_SOURCE_NAME)
        set(source_stem "${ARG_SOURCE_NAME}")
    else()
        set(source_stem "${NAME}")
    endif()

    if(ARG_BACKENDS)
        set(available_backends ${ARG_BACKENDS})
    else()
        set(available_backends CUDA OPENCL OPENCL_CPP)
    endif()

    if(USE_OPENCL_CPP)
        set(selected_backend OPENCL_CPP)
    elseif(USE_OPENCL)
        set(selected_backend OPENCL)
    else()
        set(selected_backend CUDA)
    endif()

    list(FIND available_backends "${selected_backend}" backend_index)
    if(backend_index EQUAL -1)
        message(STATUS "${NAME}: ${selected_backend} backend not enabled for this target; skipping")
        return()
    endif()

    if(selected_backend STREQUAL "CUDA")
        set(backend_source "cuda/${source_stem}.cpp")
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${backend_source}")
            message(STATUS "${NAME}: CUDA backend not implemented; skipping")
            return()
        endif()

        set(sources main.cpp ${backend_source} ${ARG_EXTRA_SOURCES})
        set_source_files_properties(${sources} PROPERTIES LANGUAGE CUDA)
        add_executable(${NAME} ${sources})
        target_link_libraries(${NAME} PRIVATE cuda_helpers)
    elseif(selected_backend STREQUAL "OPENCL_CPP")
        set(backend_source "opencl_cpp/${source_stem}.cpp")
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${backend_source}")
            message(STATUS "${NAME}: OpenCL C++ backend not implemented; skipping")
            return()
        endif()

        add_executable(${NAME} main.cpp ${backend_source} ${ARG_EXTRA_SOURCES})
        target_link_libraries(${NAME} PRIVATE opencl_cpp_helpers)
        target_compile_definitions(${NAME} PRIVATE GPU_OPENCL_CPP_BACKEND)
    else()
        set(backend_source "opencl/${source_stem}.cpp")
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${backend_source}")
            message(STATUS "${NAME}: OpenCL C backend not implemented; skipping")
            return()
        endif()

        add_executable(${NAME} main.cpp ${backend_source} ${ARG_EXTRA_SOURCES})
        target_link_libraries(${NAME} PRIVATE opencl_helpers)
        target_compile_definitions(${NAME} PRIVATE GPU_OPENCL_BACKEND)
    endif()

    if(ARG_COMPILE_DEFINITIONS)
        target_compile_definitions(${NAME} PRIVATE ${ARG_COMPILE_DEFINITIONS})
    endif()
    if(ARG_COMPILE_OPTIONS)
        target_compile_options(${NAME} PRIVATE ${ARG_COMPILE_OPTIONS})
    endif()

    if(BUILD_TESTING AND NOT ARG_NO_TEST)
        add_test(NAME ${NAME} COMMAND ${NAME})
        set_tests_properties(${NAME} PROPERTIES
            LABELS "${selected_backend}"
            TIMEOUT 300
        )
    endif()
endfunction()
