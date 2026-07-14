# OpenCL 3.1 Migration Plan

This plan tracks a staged update of the repository's OpenCL C API and OpenCL C++ wrapper backends to OpenCL 3.1 while preserving the existing backend split and correctness harnesses.

Status: planning only. The repository has not yet changed its shared OpenCL
target version to 3.1.

## References

- Khronos OpenCL 3.1 announcement: https://www.khronos.org/blog/opencl-3.1-is-here
- Khronos OpenCL registry: https://registry.khronos.org/OpenCL/
- Unified OpenCL specification: https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html

## Scope

- OpenCL C API implementations under `source/<kernel>/opencl/`.
- OpenCL C++ wrapper implementations under `source/<kernel>/opencl_cpp/`.
- Shared helpers in `source/utils/opencl_c_helpers.h` and `source/utils/opencl_helpers.h`.
- Build and documentation changes needed to expose OpenCL 3.1 support clearly.

CUDA kernels and backend-agnostic public interfaces are out of scope unless a narrow compatibility edit is required.
CUDA-only examples (`deep_learning_inference` and `fp16_dot_product`) are not
part of the OpenCL migration unless a separate task first adds an OpenCL backend.

## Migration Strategy

1. Update shared host headers first.
   - Define `CL_TARGET_OPENCL_VERSION 310` before including C API headers.
   - Change `CL_HPP_TARGET_OPENCL_VERSION` from `300` to `310` in the C++ wrapper helper.
   - Keep the minimum supported version conservative until every supported runtime is confirmed to expose OpenCL 3.1 headers and libraries.

2. Add runtime capability reporting.
   - Query and print platform name, device name, `CL_DEVICE_VERSION`, and `CL_DEVICE_OPENCL_C_VERSION` in shared helper setup paths.
   - Fail early with a clear message when an explicitly requested OpenCL 3.1 path is unavailable.
   - Keep fallback source-build paths for kernels that do not require OpenCL 3.1 features.

3. Add build-mode control.
   - Keep `clCreateProgramWithSource` as the default path for existing kernels.
   - Add an opt-in CMake flag for SPIR-V/IL ingestion experiments once the repository has a checked-in or generated IL artifact workflow.
   - Route IL program creation through shared helpers instead of duplicating loader code in each kernel.

4. Migrate kernels incrementally.
   - Start with simple elementwise kernels: `reverse_array`, `value_clipping`, `sigmoid`, `silu`, `geglu`, `swiglu`.
   - Move next to layout and bandwidth kernels: `interleave_arrays`, `rgb_to_grayscale`, `matrix_transpose`.
   - Finish with higher-risk kernels: `gemm`, `matrix_mul`, `softmax`, `prefix_sum`, `spmv`, `convolution2d`.

5. Guard optional kernel features.
   - Use `__opencl_c_*` feature macros for optional OpenCL C capabilities.
   - Do not assume sub-groups, FP16, FP64, SVM, or integer dot product support without host-side checks.
   - Provide a fallback path or clear unsupported-device error for required features.

6. Validate each phase.
   - Build the C API backend with `cmake -B build/opencl -DUSE_OPENCL=ON` and `cmake --build build/opencl -j$(nproc)`.
   - Build the C++ wrapper backend with `cmake -B build/opencl_cpp -DUSE_OPENCL_CPP=ON` and `cmake --build build/opencl_cpp -j$(nproc)`.
   - Run each changed kernel executable and compare against its existing reference path.
   - Record device, driver, OpenCL platform version, OpenCL C version, and whether fallback mode was used.

7. Update docs after implementation.
   - Update `docs/opencl-agent-guide.md` with the repository's OpenCL 3.1 policy.
   - Update `Readme.md` build notes if OpenCL 3.1 headers become a required dependency.
   - Keep migration notes in this file until all OpenCL kernels have been validated.

## Completion Criteria

- Shared C API and C++ wrapper helpers target OpenCL 3.1 headers without breaking older fallback execution for source-built kernels.
- Every implemented OpenCL backend builds in both `USE_OPENCL=ON` and
  `USE_OPENCL_CPP=ON` modes; CUDA-only examples remain cleanly excluded.
- Every migrated kernel passes its existing correctness harness.
- Any OpenCL 3.1-only code path has explicit host capability checks and a documented fallback or failure mode.
