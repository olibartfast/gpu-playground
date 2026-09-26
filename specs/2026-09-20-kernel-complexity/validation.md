# Validation specified before implementation

## Acceptance checks for future study phases

| Check | Requirements | Observable evidence |
|---|---|---|
| V-1 | R-1 | Compare root CMake targets, tracked harnesses and study index: all 18 current kernels accounted for, plus backend variants and explicit pending entries |
| V-2 | R-2 | For every study, manually trace loop bounds, launches, reductions and allocations; verify dimensions, CPU/GPU work, depth, space, traffic, and legal shape domain |
| V-3 | R-3 | Each proposed improvement names its cost category, applicability and precision constraints; verify no equation equates thread count with fixed-device speedup |
| V-4 | R-4 | Every current-algorithm claim links to source symbols; ideal alternatives and unmeasured hypotheses are labeled; verify local links resolve |
| V-5 | R-5 | Readme links reach every study; common guide and new-kernel guidance agree; `git diff --check` passes |
| V-6 | R-6 | Representative scaling reports include correctness, geometric sizes, repeated medians/spread, timing boundaries, environment, and unavailable checks |
| V-7 | R-7 | Before dispatch: declared revision, actual role configuration, protected exact acceptance command, bounded handoff and permission assessment; after attempt: single acceptance outcome and per-role ledger |

For code changes in later packets, build only affected targets for each available
backend using repository build instructions, run their correctness harnesses,
and then collect measurements. Use `benchmarkWithReset()` for output-as-input
paths. Include 1, odd sizes, B-1/B/B+1, and multiblock boundaries where legal;
test invalid/zero sizes only against a defined input contract. Reduction and
scan tolerances must reflect datatype, accumulation order and input magnitude.

## Planning evidence — 2026-09-20

- Read requested skill from the existing local checkout at
  `/home/oli/repos/skills/apply-spec-driven-development/SKILL.md`, including
  brownfield adoption and artifact templates. The supplied GitHub page could
  not be fetched by the web tool; remote/local parity was not established.
- Read repository overview, rules, agentic entrypoint and both backend guides.
- Read the local `orchestrate-ai-coding-workflows` skill and its contract and
  handoff references; incorporated it at the user's request. Remote page fetch
  failed; no remote/local parity claim. Controlled execution is not started.
- Inspected root CMake and `git ls-files 'source/*/main.cpp'`: 18 tracked
  harnesses. Planning coverage lists three pilots plus fifteen remaining names.
- Inspected prefix-sum CUDA implementation/harness and local attention scaffold.
  Their limitations are recorded as source observations in requirements.
- Future V-1–V-6 are **not completed**: full studies, builds, correctness runs,
  and scaling measurements are not part of this planning-only delivery.

Planning checks: verify packet links, all 18 names appear in the plan, and
`git diff --check`. Record results after executing these checks.

Results: Python inventory/link check passed for all 18 kernel names and local
links across seven specification files. `git diff --check` passed for tracked
changes; a separate whitespace/newline check passed for the new untracked
specification files. Existing user edits remain outside this planning change.

## Scope deviations

No implementation or performance validation was run. No changes to kernels,
existing Readme edits, or attention scaffolding are required by this packet.
No release/changelog entry is needed for an unshipped study plan.
