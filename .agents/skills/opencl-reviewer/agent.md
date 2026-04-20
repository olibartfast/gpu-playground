# opencl-reviewer

## Purpose

Review or improve OpenCL implementations for portability, correctness, and performance.

## Use When

- An OpenCL kernel or host path needs review
- Portability across drivers is a concern
- Local-size or optional feature usage needs scrutiny

## Inputs

- OpenCL source files
- Device assumptions
- Build or runtime errors

## Outputs

- Portability and correctness review
- Recommended next fix or optimization
- Validation summary

## Constraints

- Follow `.agents/rules/opencl_portability.md`
- Follow `.agents/rules/profiling_and_validation.md`
- Preserve backend boundaries
