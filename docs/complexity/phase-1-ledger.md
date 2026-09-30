# Phase 1 attempt ledger (planner-recorded)

Starting revision: `77038c4` (clean tree, master). Spec: `specs/2026-09-20-kernel-complexity/`.
Acceptance (specifier-frozen, outside worker paths): `python3 /tmp/opencode/phase1_acceptance.py`
Permission model: no harness-enforced per-worker allowlist — file ownership advisory;
reviewer rejected out-of-scope paths (none occurred). No local-inference setup; no model comparison.

| Attempt/phase | Revision | Role/model/effort | Owned paths | Acceptance command / exit | Correctness or review evidence | Tokens/context | Tool calls/turns | Wall time/cost | Interventions |
|---|---|---|---|---|---|---|---|---|---|
| P1-T2 (guide) | 77038c4 | implementer | docs/algorithmic-complexity.md | acceptance FAIL-due-to-missing-peers (peers unwritten at parallel run) | Self-check V-1..V-4 pass on own file; `git diff --check` clean | n/a (unavailable) | within 15-call budget | n/a | 0 |
| P1-T3-sigmoid | 77038c4 | implementer | docs/complexity/sigmoid.md | FAIL (own: 3 literal field names; peers missing) | `git diff --check` clean | n/a | within 15-call budget | n/a | 0 |
| P1-T3-prefix | 77038c4 | implementer | docs/complexity/prefix_sum.md | FAIL (own: 3 literal field names; peers missing) | `git diff --check` clean | n/a | within 15-call budget | n/a | 0 |
| P1-T3-gemm | 77038c4 | implementer | docs/complexity/gemm.md | FAIL (own: 1 literal field + 1 relative link; peers missing) | `git diff --check` clean | n/a | within 15-call budget | n/a | 0 |
| P1-REPAIR | 77038c4+untracked | implementer (fresh worker, corrected packet) | all 4 Phase-1 docs | **PASS** | `git diff --check` clean; no derivation changes | n/a | 6 of 10 budget | n/a | 0 (planner-issued corrected packet, no self-repair) |
| P1-review/integrate | 77038c4+untracked | planner+reviewer (this session) | Readme.md, guide table wording, this ledger | final validation below | Reviewer verdict: derivations source-grounded (see response) | n/a | — | n/a | 0 |

Notes:
- First-pass failures were checker-literal mismatches (`auxiliary space` / `memory traffic` /
  `finite-resource` keyword strings; `../source` vs `../../source` link depth), fixed by the
  repair worker without touching derivations — planner did not self-repair.
- No builds, correctness-harness runs, or GPU measurements in this phase (docs-only, per plan;
  measurement is Phase 3 T-6). No speedup claimed.
- Known risks carried forward: structural acceptance cannot validate mathematical accuracy —
  covered by reviewer verdict, not by the keyword checker alone.
