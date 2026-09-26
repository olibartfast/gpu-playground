# Requirements: per-kernel complexity study

Created: 2026-09-20. Status: planning only; no completed studies or speedups claimed.

## Goal and scope

Explain each kernel's algorithmic cost and precisely which costs GPU execution
can reduce. Big-O is a notation: not every algorithm is O(n).

- **R-1 — Coverage:** inventory every implemented `source/<kernel>` target,
  CPU reference, backend, and materially different algorithm variant. Reconcile
  CMake, source, and Readme rather than inferring support from directory names.
- **R-2 — Cost model:** every study defines input dimensions, CPU work, GPU work
  W, dependency depth S, auxiliary space, memory traffic, and finite-resource
  execution. Derive bounds from loops, reductions, launch stages, and allocations.
  Separate ideal algorithms from the implementation and its valid input range.
- **R-3 — Improvement analysis:** classify each opportunity as total-work,
  parallel-depth, memory-space, memory-traffic, or constant-factor improvement.
  Include assumptions, numerical consequences, lower bounds, and cases where
  asymptotic improvement is unavailable. Label proposals as unimplemented.
- **R-4 — Evidence:** link implementation symbols and harnesses. Distinguish
  source-derived bounds, hypotheses, and measured results. Runtime observations
  support a model; they do not prove its asymptotic bound.
- **R-5 — Discoverability:** deliver a common guide at
  `docs/algorithmic-complexity.md`, individual studies at
  `docs/complexity/<kernel>.md`, and Readme inventory links. Update the existing
  new-kernel guide to request complexity documentation alongside new kernels.
- **R-6 — Validation:** define scaling experiments before changes; require
  correctness before timing, repeatable timings, and explicit unavailable-device
  or unsupported-shape notes. No estimated speedup presented as measured.
- **R-7 — Implementation workflow:** use `orchestrate-ai-coding-workflows`
  for execution roles, bounded handoffs, permission boundaries, acceptance and
  attempt records; spec maintenance stays with `apply-spec-driven-development`.

Constraints: [mission](../mission.md), [technical boundaries](../tech-stack.md),
and existing repository rules. Preserve unrelated working-tree edits.

## Discovery evidence

- Root CMake and tracked `main.cpp` files enumerate 18 implemented harness targets.
- The local Readme adds `softmax_attention`; its untracked source contains
  placeholder functions and it has no root CMake target. Record as pending,
  excluded from implemented coverage until real code and validation exist.
- `source/prefix_sum/cuda/prefix_sum.cpp` uses Hillis-Steele stages. Its first
  stage loops to N rather than block width; the block-sum stage uses one block
  with one thread per partial sum. Inspect hardware limits before generalizing.
  A null block-sum pointer at N=256 also warrants a separate correctness task.
  These are source observations, not executed failure reports.

## Decisions and open scope

- Interpret “each” as the 18 implemented kernels under `source/`, including all
  available backends. Planned LoRA and attention get status notes, not invented
  implementation complexity claims.
- Current request authorizes planning. Kernel changes and bulk study authoring
  are future phases. Existing Readme and attention edits are preserved.
- Asked whether later optimization implementation should be scheduled; pending
  an answer, keep opportunities in the study and implementation deferred in
  roadmap phase 4. This does not block documentation planning.
- User selected `orchestrate-ai-coding-workflows` for implementation. This
  specifies the future execution method without starting kernel changes now.
- Competition submissions and external framework examples are deferred in the
  roadmap. No new benchmark framework or dependency is required for this plan.
