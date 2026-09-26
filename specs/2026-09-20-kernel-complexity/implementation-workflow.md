# Planned implementation workflow

Required skill: [orchestrate-ai-coding-workflows](https://github.com/olibartfast/skills/tree/master/orchestrate-ai-coding-workflows).
Spec ownership remains with `apply-spec-driven-development`. This document
designs execution; it does not claim a configured or completed controlled run.

## Roles and preparation

| Role | Responsibility | Routing policy |
|---|---|---|
| Specifier/reviewer | Derivations, contracts, protected acceptance criteria, final review | Strong judgement capability |
| Planner | Select a phase, isolate work, distill obligations, coordinate and log | Capable orchestration tier |
| Implementer | One bounded study batch or one kernel change | Economical tier for mechanical work; capable tier for algorithmic reasoning |

Before execution, record actual available models, reasoning effort, harness and
per-role permissions in version-controlled run configuration. Do not assume
a cheaper model is adequate for work/span proofs. Same-model delegation may
provide isolation but is not a claimed cost saving. No local provider setup or
model comparison is required for this study.

Start each controlled attempt at a declared revision in an isolated worktree
when the user's tree is dirty. Preserve the current Readme and attention work.
Put the reviewed specification and acceptance assets into that starting
revision before dispatch. Never automatically stash, reset or discard user work.

## Acceptance setup before dispatch

The specifier first implements and checks one phase-specific acceptance wrapper,
then freezes its command and assets for all attempts of that phase. Record the
exact invocation in validation.md before launching workers. The wrapper must
rebuild affected targets for code phases, run required correctness checks, and
return nonzero on failure. Documentation phases check inventory, links and
required fields, plus a protected reviewer verdict for mathematical accuracy:
a keyword checker alone cannot validate complexity derivations.

Acceptance assets and reviewer verdicts must be outside worker-writable paths.
An executable wrapper is intentionally not supplied by this planning change:
the precise study files, shape support and code targets are determined at the
phase gate. An attempt cannot start with a placeholder acceptance command.

## Handoff template

The planner fills this template separately per phase, deriving complete
obligations from requirements rather than sending workers into spec documents.

```text
Phase and attempt ID / starting revision:
Writable files: exact owned files, no others
Read-only files: source interfaces, relevant guides, protected acceptance assets
Required outcome: dimensions, derivations, evidence and backend coverage, or
                  exact kernel behavior and numerical/compatibility constraints
Permitted commands: exact reads, targeted builds/checks, final acceptance command
Permission enforcement: enforced controls and advisory controls listed separately
Working method: read existing files; write complete final files; preserve others' work
Budget: recorded step/time ceiling and planner stop responsibility
Acceptance: exact frozen command; run once as the final attempt action
Stop: return verdict and evidence, pass or fail; no edits after acceptance
```

Workers are not alone in the repository: assign disjoint files and do not revert
others' changes. Source and mathematical review can stay in one capable context;
delegate only sufficiently bounded work. Targeted checks may run during development.
After a failed acceptance attempt, the planner issues a corrected packet to a
fresh worker; it does not silently repair a delegated implementation itself.

## Permission boundaries

The current session enforces broad filesystem boundaries, not a separate
per-worker file allowlist. Thus exact file ownership and step limits are
currently advisory, not security guarantees. Before controlled execution,
configure path/tool restrictions where supported; otherwise explicitly record
the limitation, isolate the checkout, keep acceptance outside its writable
surface, and reject unexpected changed paths during review. Give reviewers a
read-only execution configuration. A shared worktree alone is not isolation.

## Attempt ledger template

Record one row per attempt, including failures. Store ledger updates through
the planner after receiving the worker's final result.

| Attempt/phase | Revision | Role/model/effort | Owned paths | Acceptance command / exit | Correctness or review evidence | Tokens/context | Tool calls/turns | Wall time/cost | Interventions |
|---|---|---|---|---|---|---|---|---|---|
| Not started | — | — | — | — | — | — | — | — | — |

Measure planner and worker separately; mark unavailable metrics unavailable.
Agent cost is separate from GPU benchmark performance. If configurations are
compared later, hold task, revision, acceptance and prompt structure fixed,
change one variable, pin reasoning effort, and repeat each configuration.

Gate: R-7/V-7. Known risks are incorrect derivations passing structural checks,
advisory permissions, planner self-repair, and context/cost shifted to planning.
