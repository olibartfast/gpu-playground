# The OpenCL implementation is not wired up yet; force the CUDA backend so an
# OpenCL configure skips this target cleanly instead of failing on missing files.
#
# No CTest entry: the harness needs a user-supplied RGBA image argument. The
# benchmark mode is not a pass/fail correctness check.
gpu_add_kernel(deep_learning_inference
    SOURCE_NAME dnn_inference
    BACKENDS CUDA
    EXTRA_SOURCES cuda/dnn_inference_optimized.cpp
    COMPILE_DEFINITIONS "WEIGHTS_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/weights.bin\""
    COMPILE_OPTIONS
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-use_fast_math;-lineinfo>"
        "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>"
    NO_TEST
)
