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
- An updated `Readme.md` kernel inventory and usage entry
- Updated supporting docs when structure or workflow changes

## Constraints

- Follow `.agents/rules/backend_boundaries.md`
- Follow `.agents/rules/kernel_harness_contract.md`
- Treat the `Readme.md` update as part of the kernel implementation, not an
  optional follow-up
- Do not change unrelated kernels
