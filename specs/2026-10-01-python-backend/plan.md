# Plan: Python DSL backends

Requirements: [requirements.md](requirements.md). Validation is defined in
[validation.md](validation.md) before implementation. Execution follows
[implementation-workflow.md](implementation-workflow.md).

## Phase 0 — Acceptance assets (specifier, before dispatch)

- **T-0:** freeze `acceptance/check.sh` and `acceptance/helper_check.py`, and
  record the invocation in validation.md. Workers never edit `acceptance/`.

## Phase 1 — Helper (walk)

- **T-1 (R-2, R-6):** write `source/utils/python/gpu_bench.py` and
  `source/utils/python/requirements.txt`.

Gate: V-1 (helper unit checks pass on CUDA).

## Phase 2 — Migrate scripts and document (run)

- **T-2 (R-3, R-4, R-5):** owns the three scripts in F-1. Do Triton sigmoid
  first as the reference migration, then the two vector-add scripts.
- **T-3 (R-7):** owns AGENTS.md, Readme.md, docs/adding-a-new-kernel.md and
  specs/tech-stack.md. It can run in parallel with T-2 because the API is frozen
  in R-2 and the file sets are disjoint.

Gate: V-2–V-6, then full acceptance and V-7 review.

## Integration

- One PR from `feat/python-backend` containing this packet, the code and the docs.
- On merge, update the roadmap and record evidence.
- The other two branches (PR #6, PR #7) are already merged; this branch is rebased onto them.
