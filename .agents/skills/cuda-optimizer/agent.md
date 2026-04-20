# cuda-optimizer

## Purpose

Optimize CUDA kernels with a measurement-first workflow.

## Use When

- A CUDA kernel is slow
- Profiling points to a CUDA-side bottleneck
- A CUDA implementation needs tuning or review

## Inputs

- Target CUDA source files
- Baseline behavior or measurements
- Relevant correctness expectations

## Outputs

- Bottleneck hypothesis
- Recommended or implemented optimization
- Validation summary

## Constraints

- Follow `.agents/rules/cuda_optimization.md`
- Follow `.agents/rules/profiling_and_validation.md`
- Preserve correctness and backend boundaries
