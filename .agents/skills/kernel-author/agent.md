# kernel-author

## Purpose

Add or refactor kernels while preserving the repository's backend structure and harness conventions.

## Use When

- Adding a new kernel
- Restructuring an existing kernel directory
- Updating `CMakeLists.txt` and harness wiring for a kernel

## Inputs

- Target kernel name
- Existing repo layout
- Requested backend scope

## Outputs

- A consistent kernel directory
- Correct build wiring
- Updated docs when structure changes

## Constraints

- Follow `.agents/rules/backend_boundaries.md`
- Follow `.agents/rules/kernel_harness_contract.md`
- Do not change unrelated kernels
