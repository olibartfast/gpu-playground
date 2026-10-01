# Implementation workflow

Executes [plan.md](plan.md) with `orchestrate-ai-coding-workflows`. Spec
maintenance stays with `apply-spec-driven-development`, as in the
[kernel-complexity workflow](../2026-09-20-kernel-complexity/implementation-workflow.md).

## Roles

| Role | Packet | Writable paths |
|---|---|---|
| Specifier / orchestrator (main session) | T-0, dispatch, ledger, spec upkeep | `specs/2026-10-01-benchmark-device-timing/` |
| Implementer | T-1 | `source/utils/benchmark.h` |
| Implementer | T-2 | `source/utils/benchmark_helpers.h`, `source/fp16_dot_product/main.cpp`, `source/categorical_cross_entropy/main.cpp`, `source/gaussian_blur/main.cpp` |
| Implementer | T-3 | `AGENTS.md`, `Readme.md`, `docs/EXAMPLES.md`, `docs/cuda-agent-guide.md`, `docs/adding-a-new-kernel.md`, `specs/tech-stack.md` |
| Reviewer (read-only) | V-8 | none |

Order: T-0 → T-1 → (T-2 ∥ T-3) → acceptance → V-8. Each handoff inlines its
obligations from requirements.md. Workers do not edit `acceptance/`, do not
commit, push or switch branches, and end by running the frozen acceptance
command once, reporting pass or fail verbatim. After a failed attempt, the
orchestrator issues a corrected packet to a fresh worker instead of repairing it itself.

## Permission boundaries

File ownership is advisory: the session enforces no per-worker path allowlist.
Mitigations: disjoint ownership, a read-only reviewer, and rejecting unexpected
changed paths at review. Only T-1 and T-2 build, and they never run
concurrently, so the shared `build/default` tree is never built by two workers at once.

## Attempt ledger

| Attempt | Packet | Revision | Role/model | Acceptance exit | Evidence | Interventions |
|---|---|---|---|---|---|---|
| 0 | T-1, T-3 | pre-spec | implementer | not run | stopped; edits stashed unreviewed (superseded) | user halted: spec first |
| 1 | T-1 | 49b3ce4 | implementer | phase gate V-1..V-4 pass | summarize_check PASSED; build ok; 4 harnesses ok | none |
| 1 | T-2 | 49b3ce4 + T-1 | implementer | check.sh exit 0 | full acceptance | hit turn limit while acceptance ran; finished on its own |
| 1 | T-3 | 49b3ce4 + T-1 | implementer | own checks pass | grep/diff-check/API names | resumed twice at its turn limit |
| 1 | V-8 | working tree | reviewer | APPROVE | N-1..N-6 non-blocking | — |
| 2 | N-1..N-3 fix | working tree | implementer (fresh) | check.sh exit 0 (orchestrator re-run) | full acceptance on final tree | hit turn limit before its checks; orchestrator ran them |
