# Algorithmic Complexity Studies: Shared Guide and Kernel Inventory

Status: planning-phase guide for the [kernel-complexity
spec](../specs/2026-09-20-kernel-complexity/requirements.md). This document
defines the shared cost model, vocabulary, improvement taxonomy, evidence
rules, and per-kernel study template, and it reconciles the current kernel
inventory. **No per-kernel studies, speedups, or measured results are claimed
by this guide.** Individual studies live under `docs/complexity/<kernel>.md`
(phase 2 deliverable); entries below marked "no study yet" have not been
written, and none of their complexity numbers are asserted here.

## 1. Vocabulary: Big-O is a notation, not every algorithm is O(n)

Before assigning any bound, a study must name its **operation** and
**dimensions**. Common dimension symbols used across studies:

- `n` — element count for flat elementwise arrays.
- `M, N, K` — dense matrix-multiply dimensions (`C[M×N] = A[M×K] · B[K×N]`).
- `H, W, C_{in}, C_{out}` — image height/width and channels (convolution,
  inference).
- `nnz` — number of stored nonzeros for sparse formats (SpMV); row length
  imbalance is a separate dimension.
- `rows` / `classes` — reduction extent for softmax rows, cross-entropy
  classes, dot-product length.
- `B` — a chosen implementation block width (block size), **not** a
  complexity parameter. Textbook bounds derived at a fixed `B` must not be
  assigned blindly to an implementation unless its `B`-dependent derivation
  has been written out (see the scan pilot, §3.2).

An algorithm is O(n), O(n log n), O(n²), or O(nnz) only over a stated legal
input domain. Fixed-size, fixed-loop implementations are Θ(1) with respect to
inputs they never read and Θ(min(n, cap)) as soon as a cap appears in the
code — studies must derive bounds from actual loops, not from names.

## 2. Source-link convention

Every claim about the current implementation links to a tracked symbol:

- Implementation files: `source/<kernel>/cuda/<kernel>.cpp`,
  `source/<kernel>/opencl/…`, `source/<kernel>/opencl_cpp/…`.
- Reference/harness: `source/<kernel>/main.cpp` and the CPU function inside
  the implementation file (e.g. `sigmoid_cpu` in
  [`source/sigmoid/cuda/sigmoid.cpp`](../source/sigmoid/cuda/sigmoid.cpp)).
- CMake targets for backend coverage: root `CMakeLists.txt` plus each
  `source/<kernel>/CMakeLists.txt`.

Rule (R-4): no equation or bound appears in a study without a source anchor,
and ideal alternatives are cited as design references, not as links implying
they exist in this repo.

## 3. Cost model

### 3.1 Core quantities (R-2)

Each study defines, for one named backend variant:

| Quantity | Meaning | Derived from |
|---|---|---|
| Dimensions | Input shape(s) and legal domain (what sizes compile/run) | Kernel signatures, launch code, harness |
| CPU work | Arithmetic and passes in the reference routine | Reference loops |
| GPU work `W` | Total primitive arithmetic/memory operations executed across **all** threads/stages (parallelism-independent total) | Sum over innermost loops of all kernels and stages |
| Dependency depth `S` | Critical-path length: t steps that cannot overlap because each depends on the previous | Reduction/scan stage counts, serialized loops |
| Auxiliary space | Shared memory, block-sum buffers, workspace, intermediates | Allocations and `__shared__` declarations |
| Memory traffic | Reads + writes to global memory (minimum, plus what the implementation actually does: tile loads, re-reads, atomics) | Innermost access pattern |

### 3.2 Finite-device model

The ideal lower bound on runtime for a fixed device with `P` execution
resources (SIMT lanes / SM cores, i.e. a **model of the hardware**, NOT the
block grid or the number of launched threads):

```
T_P  >=  max( W/P , S )
```

- `W/P`: even perfectly parallel work needs at least `W/P` steps.
- `S`: a dependency chain cannot finish faster than its depth no matter how
  many resources are idle.

Both terms are lower bounds: the bound is `max`, not a sum.

**Assumptions before using the scheduling estimate `O(W/P + S)`** — it holds
only if: (a) work is composable into unit operations schedulable on `P`
resources (Brent's scheduling assumptions); (b) no stage is left serialized
by a lock/atomic/load imbalance; (c) memory, not compute, is not the binding
resource; (d) overheads (launch, sync, transfer) are negligible at the sizes
measured. A study states these assumptions explicitly; if any fails, the
estimate is a hypothesis, not the model.

**What `P` is not:** `P` is never the number of threads launched. Launching
1000× more threads does not create 1000× more hardware; for fixed `P`,
increasing `n` generally keeps the same asymptotic work growth, adding
constant factors (and wave quantization) rather than changing the Big-O in
`n`. No study may write an equation equating thread count with fixed-device
speedup.

### 3.3 Caveats between the ideal bound and observed time

Actual time additionally depends on:

- **Bandwidth**: achievable GB/s vs the infinite-bandwidth implicitly assumed
  by `W/P`; bandwidth-bound kernels scale with bytes/element, not FLOPs.
- **Host↔device transfers**: PCIe traffic is frequently the dominant term for
  small `n` and counts in the traffic column.
- **Synchronization**: `__syncthreads()` between tile loads, stage-separated
  kernel launches for multi-pass scans.
- **Front-end constants and launches**: kernel launch latency and per-block
  scheduling are Θ(number of launches), independent of `n` — these dominate
  small sizes and floor the curve.
- **Occupancy and resource limits**: register/shared-memory usage caps resident
  blocks; wave quantization makes timing step-wise in `n`; occupancy below one
  wave fails assumption (b) of the Brent estimate.

A study separates these regimes explicitly; a runtime curve in one regime
does not certify the asymptotic bound.

## 4. Improvement taxonomy (R-3)

Each opportunity in a study is tagged with exactly one primary category, with
applicability, numerical consequences, and any lower bound:

1. **Total-work** — reduce `W` (e.g. Hillis-Steele `O(n log n)` scan →
   work-efficient `O(n)` scan; restructuring double passes). Changing the
   algorithm's asymptotic work requires a different input contract or a
   fundamentally different scheme; identical I/O contract usually pins `W`
   from below (must read all inputs, write all outputs).
2. **Parallel-depth** — reduce `S` without changing `W` (e.g. serial per-row
   K loop → parallel tree reduction over K for GEMM accumulators). Improves
   the `S` term of `T_P`; useless where `W/P` dominates.
3. **Memory-space** — reduce auxiliary storage (e.g. avoid materializing an
   intermediate block-sum or score buffer). Overlaps with traffic when the
   extra space is round-tripped through memory.
4. **Memory-traffic** — reduce bytes moved (tiling/shared-memory reuse,
   fusion, vectorized access). Ordinary tiling of GEMM does **not** change the
   `O(MNK)` arithmetic count; it changes reuse/traffic and constants.
5. **Constant-factor** — same asymptotics, smaller constants: `__expf`
   intrinsics, float4 vectorization, fewer branches. Always quantify where the
   constant comes from in the source (intrinsic choice, uncoalesced stride,
   branchy tail).

Every taxonomy entry also lists: assumptions (shape legality, precision),
numerical consequences (accumulation order changes, fast-math error), the
attainable lower bound, and cases where **asymptotic improvement is
unavailable** under the fixed dense I/O contract. All proposals are labeled
**unimplemented / hypothesis** until code, correctness, and measurement exist.

## 5. Correctness constraints (for any future change)

- Correctness is validated before timing; harnesses compare against the CPU
  reference within stated tolerance (~1e-4f).
- Reduction re-association (parallel scans, tree reductions) changes
  accumulation order; tolerance must account for datatype, order, and input
  magnitude.
- Fast-math intrinsics (`__expf`) are approximations: state the precision
  change, do not silently swap.
- Odd sizes, `B−1/B/B+1`, 1-element, and block-boundary shapes must be legal
  in the harness or explicitly out of contract; tails (non-multiple-of-4
  vectorized loads) deserve an explicit correctness note.
- Zero/invalid sizes are tested only against a defined input contract.
- Known source observations logged in the spec (e.g. the prefix-sum first
  stage looping to `n` rather than block width, and the single-block
  block-sum stage) are correctness flags to resolve before any optimization.

## 6. Evidence vs hypothesis vs measured (R-4, R-6)

Three tiers, and studies label every claim:

- **Source-derived bound (evidence of code, not of hardware)** — traced from
  loops, stages, allocations in linked source.
- **Hypothesis** — a model prediction about hardware behavior
  (e.g. "bandwidth-bound above n≈…" or "occupancy caps at …"); must state its
  assumptions and how it would be falsified.
- **Measured** — `benchmark()`/`benchmarkWithReset()` medians and spread over
  geometric size sweeps, with environment, timing boundary, and tolerance
  recorded per R-6.

Runtime measurements **support** a model; they never **prove** its Big-O. A
straight line on a log-log plot is consistent with Θ(n) over the measured
range only.

## 7. Per-kernel study template

Every study in `docs/complexity/<kernel>.md` uses exactly these sections:

1. **Operation and dimensions** — what it computes; symbol/shape vocabulary;
   legal input domain.
2. **Source links** — implementation symbols, CPU reference, harness,
   CMake targets, per available backend.
3. **CPU work** — reference passes and their costs.
4. **Actual GPU work / depth** — `W` and `S` for each implemented variant,
   derived stage by stage (this is the *implemented* algorithm, not the
   textbook ideal).
5. **Auxiliary storage and traffic** — shared memory, buffers, intermediates;
   global bytes read/written per call.
6. **Finite-device model** — `T_P >= max(W/P, S)`, the stated assumptions for
   using `O(W/P + S)`, and expected regime (bandwidth/compute/latency).
7. **Improvement options** — taxonomy-tagged list per §4, with assumptions,
   numerical consequences, lower bounds, unavailability notes; all labeled
   unimplemented/hypothesis.
8. **Correctness constraints** — tolerance, reorder effects, shape edge
   cases, invalid inputs.
9. **Evidence and limitations** — which claims are source-derived,
   hypothetical, or measured; missing hardware/instrumentation noted.

## 8. Kernel inventory and backend coverage (R-1)

Root CMake enumerates **18 implemented harness targets**. Backend coverage
summary verified against directory/source structure:

- 15 kernels have all three backends (cuda + opencl + opencl_cpp).
- `deep_learning_inference` has cuda + opencl, **no opencl_cpp**.
- `categorical_cross_entropy` and `fp16_dot_product` are **CUDA-only**.
- Pending (NOT counted as implemented; no study may invent complexity for
  them): `lora_linear` — empty `source/lora_linear/` directory, no harness,
  no CMake target; `vector_addition` — Python-only
  (`source/vector_addition/`), no C++ harness, no CMake target.

| Kernel | CUDA | OpenCL | OpenCL C++ | Study | Variants noted in source |
|---|---|---|---|---|---|
| categorical_cross_entropy | yes | — | — | no study yet | CUDA-only |
| convolution2d | yes | yes | yes | no study yet | |
| deep_learning_inference | yes | yes | — | no study yet | no opencl_cpp |
| fp16_dot_product | yes | — | — | no study yet | CUDA-only |
| geglu | yes | yes | yes | no study yet | |
| gemm | yes | yes | yes | phase-1 pilot ([study](complexity/gemm.md)) | tiled `gemmTiled`, 2×2-register `gemmTiled2x2` |
| interleave_arrays | yes | yes | yes | no study yet | |
| matrix_mul | yes | yes | yes | no study yet | |
| matrix_transpose | yes | yes | yes | no study yet | |
| prefix_sum | yes | yes | yes | phase-1 pilot ([study](complexity/prefix_sum.md)) | Hillis-Steele blocked scan (see §3 pilot notes) |
| reverse_array | yes | yes | yes | no study yet | |
| rgb_to_grayscale | yes | yes | yes | no study yet | |
| sigmoid | yes | yes | yes | phase-1 pilot ([study](complexity/sigmoid.md)) | scalar `sigmoid_kernel`, vectorized `sigmoid_kernel2` |
| silu | yes | yes | yes | no study yet | |
| softmax | yes | yes | yes | no study yet | |
| spmv | yes | yes | yes | no study yet | |
| swiglu | yes | yes | yes | no study yet | |
| value_clipping | yes | yes | yes | no study yet | |
| lora_linear | pending | pending | pending | no study yet | empty dir; no harness/CMake |
| vector_addition | pending | pending | pending | no study yet | Python-only; no C++ harness/CMake |

("phase-1 pilot" means the pilot study linked in that row, delivered with this
phase per the [plan](../specs/2026-09-20-kernel-complexity/plan.md); it does not
prestate any numbers beyond what the linked study derives.)

## 9. Pilot teaching examples (checked against the pilots, in spirit verbatim)

These three examples anchor the model. They are teaching derivations; the
actual coded variants are analyzed against them in the pilot studies.

### 9.1 Elementwise (sigmoid) — linear work stays linear; vectorization changes constants

A pointwise function `y[i] = 1/(1+e^{-x[i])}` performs Θ(1) work per element,
so `W = Θ(n)`, `S = Θ(1)` (each output depends only on its input), and the
ideal `T_P = max(Θ(n/P), Θ(1))`. **Linear work remains linear**: no
restructuring of elementwise code changes the Big-O; the only asymptotic floor
besides `n` is reading `n` inputs and writing `n` outputs. Vectorization —
`float4` loads/stores, four elements per thread with `__expf`
([`sigmoid_kernel2`](../source/sigmoid/cuda/sigmoid.cpp)) — **changes
constants** (fewer memory transactions, cheaper exponentials) and changes
numerics (`__expf` vs `expf` precision), not the asymptotic class. Its
scalar tail loop keeps correctness at non-multiple-of-4 sizes.

### 9.2 Scan (prefix_sum) — work/depth trade-off, and derive the blocked version, don't borrow its bound

Two ideal schemes for the inclusive prefix sum over `n` elements:

- **Serial** scan: `W = Θ(n)`, `S = Θ(n)` — no parallelism without rework.
- **Hillis–Steele** (doubling offsets `offset *= 2`,
  [`block_inclusive_scan`](../source/prefix_sum/cuda/prefix_sum.cpp)): each
  of `n` lanes does `log2 n` steps → **work `W = Θ(n log n)`**, **depth
  `S = Θ(log n)`**. More total work, far shorter critical path.
- **Work-efficient** scan (Blelloch-style up/down sweep): **`W = Θ(n)`**,
  **`S = Θ(log n)`** — both `W` and `S` ideal for this operation; both
  doubling and work-efficient schemes reach ideal `O(log n)` depth, so the
  real difference is the work term.

**Deriving the existing blocked implementation separately:** the CUDA code is
a three-stage blocked scheme: per-block inclusive scan (`block_inclusive_scan`,
a doubling scan over a block of width `B`), a scan of the `ceil(n/B)` block
sums (`scan_block_sums`), then `add_block_sums` broadcasting each scanned
block sum to the block's elements. Its costs follow from **`B` and
`ceil(n/B)`**, from the loops actually written — e.g. ≤ `n·log2 B` +
`ceil(n/B)·log2(ceil(n/B))` + `n` steps with a launch-heavy depth of
`log2 B + log2(ceil(n/B)) + 2` kernel stages — and must be derived from that
`B`-dependent structure. Do **not** assign the textbook Θ(n log n) or Θ(n)
bound to the blocked implementation by analogy; with finite `B` set by
hardware, its stage count, launch count, and auxiliary `ceil(n/B)` space are
properties of this specific code. Treat hardware limits (`n` vs block width,
single-block block-sum stage) as open correctness/scaling questions flagged
in the spec, bullets under "Discovery evidence".

### 9.3 GEMM (gemm) — O(MNK) persists under tiling; work can change only if the input contract supplies structure

Classical dense `C[M×N] = A[M×K]·B[K×N]` requires `M·N·K` multiply-adds:
every output element touches every element of its `K`-dimension strip, so
**`W = Θ(MNK)`, `S = Θ(K)` for a serial per-output K loop** (one thread
accumulates a scalar over K). Tiling with shared memory
([`gemmTiled`](../source/gemm/cuda/gemm.cpp), 2×2 register variant
`gemmTiled2x2`) does **not** change this: each output still consumes K
products, so **`O(MNK)` arithmetic persists under ordinary tiling — tiling
changes reuse and traffic** (each A/B element is read `O(N/TILE_K)` /
`O(M/TILE_K)` times instead of being re-fetched per output; traffic drops
from Θ(MNK·reuse_factor) toward Θ(MNK/max(M,N,TILE)+M·K+K·N)) and launches
synchronization (`__syncthreads()` per tile).

A **parallel reduction over K** per output (proposed, unimplemented) would
change `S` from Θ(K) to Θ(log K) while keeping `W = Θ(MNK)` — a depth-term
improvement, not a work-term one; it adds auxiliary partial sums and a
re-association of the accumulation order.

**The work itself changes only if the input contract supplies structure:**
sparsity (A or B sparse → Θ(nnz) per output strip), low rank (`K` effective),
diagonal/banded structure, or symmetric batch inputs can reduce the required
multiply-adds below Θ(MNK) — but only because the *new contract* makes whole
products guaranteed-zero or shared. A dense in/out contract pins Ω(MN) reads
of result-touching inputs and Ω(MNK) arithmetic from below. Also note the
serial-K/parallel-K choice only affects `T_P` when `M·N·P-ratio` supplies
more parallelism than the `M·N` outputs already do; state which regime
applies (W-bound vs S-bound) rather than assuming depth wins.

## 10. Coverage obligations going forward

- Phase 2 writes one study per implemented kernel, batched per the
  [plan](../specs/2026-09-20-kernel-complexity/plan.md); each study follows
  §7 exactly and links from Readme once phase 3 lands those links.
- Pending entries (`lora_linear`, `vector_addition`) get status notes only —
  never invented implementation complexity or speedup claims.
- Every new kernel added by [`docs/adding-a-new-kernel.md`](adding-a-new-kernel.md)
  must produce a complexity study with its change (per R-5).
- Unimplemented ideas, per §4/§6, stay labeled hypothesis until measured —
  no estimated speedup may ever be presented as measured.
