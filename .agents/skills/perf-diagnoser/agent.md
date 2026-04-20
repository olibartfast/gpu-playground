# perf-diagnoser

## Purpose

Classify the dominant bottleneck before deeper optimization work starts.

## Use When

- A kernel regressed
- A profiler trace needs interpretation
- The right optimization direction is unclear

## Inputs

- Relevant source files
- Timing output or profiler data
- Observed symptoms

## Outputs

- Bottleneck classification
- Evidence
- Best next measurement or change

## Constraints

- Prefer diagnosis over speculation
- Do not jump straight to broad rewrites
- Cite the relevant rule files consulted
