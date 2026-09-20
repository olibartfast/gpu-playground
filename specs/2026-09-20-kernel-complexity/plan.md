# Plan: study algorithmic complexity across kernels

This is a proposed sequence, not a report of completed implementation.
Requirements: [requirements.md](requirements.md).
Checks are defined in [validation.md](validation.md) before implementation.
Execute future phases through [implementation-workflow.md](implementation-workflow.md),
using the user-selected `orchestrate-ai-coding-workflows` skill.

## Phase 1 — Establish the model with three pilots

1. **T-1 (R-1):** reconcile CMake targets, tracked sources, CPU references,
   backend coverage, algorithm variants, and supported shapes. Record missing
   coverage or suspected correctness defects separately.
2. **T-2 (R-2, R-3):** write the shared guide and per-kernel study template:
   operation and dimensions; source links; CPU work; actual GPU work/depth;
   auxiliary storage and traffic; finite-device model; improvement options;
   correctness constraints; evidence and limitations.
3. **T-3 (R-2–R-4):** study `sigmoid`, `prefix_sum`, and `gemm` first.
   These exercise elementwise, communicating, and compute-heavy algorithms.

Use T_P >= max(W/P, S) as an ideal lower bound, where P is a model of available
execution resources, not the number of launched threads. Explain assumptions
before using the ideal work-span scheduling estimate O(W/P + S). Actual time
also depends on bandwidth, transfers, synchronization, launches, and occupancy.
With fixed P, increasing n generally retains the same asymptotic work growth.

Pilot teaching examples, to be checked against every implementation:

- Elementwise processing: linear work remains linear; independent outputs
  expose parallelism. Vectorization usually changes constants.
- Scan: compare global Hillis-Steele O(n log n) work with a work-efficient
  O(n) scan, both with ideal O(log n) depth. Derive the existing blocked
  implementation separately using block width B and ceil(n/B) partial sums;
  do not blindly assign the textbook bound to it.
- Classical GEMM: O(MNK) arithmetic persists with ordinary tiling; tiling
  changes reuse/traffic. Distinguish serial per-output K loops from a proposed
  parallel reduction. Special matrix structure can change required work only
  if the input contract supplies that structure.

Gate: V-1–V-4 on the pilots. Resolve the template before expanding coverage.

## Phase 2 — Complete the inventory in small batches

Each name below receives its own study. Shared derivations may be linked, while
backend-specific differences remain explicit. These are questions to answer,
not prevalidated complexity assignments.

| Batch | Kernels | Main analysis questions |
|---|---|---|
| A | `silu`, `geglu`, `swiglu`, `value_clipping`, `rgb_to_grayscale` | Actual element count, gating semantics, scalar/vector paths, minimum reads/writes |
| B | `reverse_array`, `interleave_arrays`, `matrix_transpose` | Permutation work, in-place space, layout, coalescing and tiling |
| C | `softmax`, `fp16_dot_product`, `categorical_cross_entropy` | Local/global reduction stages, serial tails, rows/classes, precision, intermediates |
| D | `matrix_mul`, `spmv` | Dense dimensions versus nnz and row lengths, all sparse variants, load imbalance and format conversion |
| E | `convolution2d`, `deep_learning_inference` | Image/filter/channel dimensions, layer-by-layer costs, fixed versus variable sizes, workspace and fusion |

**T-4 (R-1–R-4):** inspect each harness and available backend implementation,
derive costs, and explain improvement options. For direct convolution,
separability/FFT alternatives require explicit applicability and numerical
discussion; never imply they are already implemented.

**T-5 (R-3):** rank candidates by what changes: arithmetic work, critical path,
space, traffic, or constants. Note hard limits such as reading all general
inputs and producing every required output. Attention may have a future note
on avoiding a materialized score matrix, clearly separate from runtime claims
and from changing dense attention's arithmetic complexity.

Gate after each batch: V-1–V-4. Publish accurate individual studies without
waiting for every batch or for GPU hardware access.

## Phase 3 — Connect theory to measurements and navigation

**T-6 (R-4, R-6):** specify geometric size sweeps for the three pilots and at
least one reduction, permutation, sparse, and convolution example. Vary each
meaningful dimension independently; for SpMV vary nnz and row imbalance.
Use tiny, odd, block-boundary, and larger legal shapes. Add minimal harness
parameters only in a separately scoped change if current harnesses fix sizes.

Capture CPU and GPU medians/spread using existing benchmark helpers. Record
end-to-end and kernel-only boundaries where supported, synchronization,
allocation/build costs, device/compiler/backend, precision, and validation
tolerance. Report unavailable instrumentation or hardware explicitly.
Plot runtime versus size and suitable normalized costs; identify overhead,
bandwidth, compute, and resource-limit regimes without claiming plots prove Big-O.

**T-7 (R-5):** link studies from Readme; update `docs/adding-a-new-kernel.md`
and the agentic entrypoint so future kernels maintain coverage. Preserve the
user's existing Readme edits. Mark roadmap completion only against evidence.

Gate: V-1–V-6, with measured and source-only coverage distinguished.

## Phase 4 — Follow-up implementation, deferred

Select one supported opportunity after study: first resolve any correctness
blocker, then form an optimization hypothesis. Create a separate feature packet
with baseline, exact input contract, expected cost change, correctness cases,
and measurement plan. Compare the same shapes, hardware and timing boundaries.
No optimization is selected or speedup promised by this planning change.

Replan after every gate if actual backend behavior contradicts a proposed
model. Update requirements, plan, and evidence together; keep failed hypotheses.
