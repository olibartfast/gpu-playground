# Agent Rule: OpenCL Portability

## Purpose

Keep OpenCL changes portable across drivers and device classes while still allowing optimization.

## Rules

- Query optional device capabilities instead of assuming they exist.
- Guard optional OpenCL C features with `__opencl_c_*` checks.
- Keep fallback paths clear when sub-groups, FP16, or other optional features are absent.
- Prefer simple coalesced access before adding `__local` tiling.
- Re-validate local-size choices on the actual target device class.
- Always inspect and surface build logs when kernel compilation fails.

## Review Questions

1. Does the code assume an optional OpenCL feature without checking it?
2. Is the local-size choice justified?
3. Will this still make sense on a different driver or vendor stack?
