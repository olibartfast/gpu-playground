# Implementation workflow

Executes [plan.md](plan.md) with `orchestrate-ai-coding-workflows`. Spec upkeep
stays with `apply-spec-driven-development`.

## Roles

| Role | Packet | Writable paths |
|---|---|---|
| Specifier / orchestrator (main session) | T-0, dispatch, ledger, spec upkeep | `specs/2026-10-01-python-backend/`, `specs/roadmap.md` |
| Implementer | T-1 | `source/utils/python/gpu_bench.py`, `source/utils/python/requirements.txt` |
| Implementer | T-2 | `source/sigmoid/python/triton/sigmoid.py`, `source/vector_addition/python/triton/vector_addition.py`, `source/vector_addition/python/cute-dsl/vector_addition.py` |
| Implementer | T-3 | `AGENTS.md`, `Readme.md`, `docs/adding-a-new-kernel.md`, `specs/tech-stack.md` |
| Reviewer (read-only) | V-7 | none |

Order: T-0 → T-1 → (T-2 ∥ T-3) → acceptance → V-7.

Rules for workers:
- Each handoff inlines its obligations from requirements.md, including the frozen R-2 API.
- Workers run in the `../gpu_playground-python` worktree.
- They do not edit `acceptance/`, and they do not commit, push or switch branches.
- They end by running the frozen acceptance once and reporting the outcome verbatim.

A failed attempt gets a corrected packet sent to a fresh worker.

## Permission boundaries

Ownership is advisory: there is no per-worker path allowlist. Mitigations are
disjoint file sets, a read-only reviewer, and rejecting unexpected paths in review.
GPU use is light (seconds per script), so T-2 can share the GPU with nothing else running.

## Attempt ledger

| Attempt | Packet | Revision | Role/model | Acceptance exit | Evidence | Interventions |
|---|---|---|---|---|---|---|
| 1 | T-1 | 310b9ea | implementer | V-1 pass | helper_check PASSED | none |
| 1 | T-2 | 310b9ea + T-1 | implementer | check.sh exit 0 | V-1..V-6, V-4 GAP | none |
| 1 | T-3 | 310b9ea + T-1 | implementer | own checks pass | V-6 greps | resumed once at its turn limit |
| 1 | V-7 | working tree | reviewer | REJECT | check_close raised/broadcast; CuTe timed JIT; doc duplication | specifier extended helper_check.py |
| 2 | A, B, C fixes | working tree | 3 fresh implementers | check.sh exit 0 (orchestrator re-run) | full acceptance | C resumed once at its turn limit |
| 2 | V-7 | working tree | reviewer | APPROVE | one accepted nit | — |
