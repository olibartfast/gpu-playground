# Algorithmic Complexity Studies: Shared Guide and Kernel Inventory

> Audience: humans, agents · Applies to: all · Last reviewed: 2026-10

This guide supports the [kernel-complexity
specification](../../specs/2026-09-20-kernel-complexity/requirements.md). It
defines the shared cost model, vocabulary, improvement taxonomy, and evidence
rules used by every per-kernel study, and it reconciles the current kernel
inventory.

The guide is a planning document. It claims no per-kernel studies, speedups, or
measured results. Individual studies live under `docs/complexity/<kernel>.md`
and are delivered in Phase 2. Entries marked "no study yet" have not been
written, and this guide asserts no complexity numbers for them.

## 1. Vocabulary: Big-O Is a Notation, Not a Guarantee

Before assigning any bound, a study must name its **operation** and its
**dimensions**. These are the dimension symbols used across studies:

- `n` — element count for flat elementwise arrays.
- `M, N, K` — dense matrix-multiply dimensions
  (`C[M×N] = A[M×K] · B[K×N]`).
- `H, W, C_{in}, C_{out}` — image height, width, and channels (convolution and
  inference).
- `nnz` — number of stored nonzeros for sparse formats (SpMV). Row-length
  imbalance is a separate dimension.
- `rows` / `classes` — reduction extent for softmax rows, cross-entropy classes,
  and dot-product length.
- `B` — a chosen implementation block width, **not** a complexity parameter.
  Textbook bounds derived at a fixed `B` must not be assigned blindly to an
  implementation unless its `B`-dependent derivation has been written out (see
  the scan pilot in §9.2).

`O(...)` denotes an asymptotic upper bound; `Theta(...)` denotes a tight bound
that is both an upper and a lower bound up to constant factors. An algorithm is
`O(n)`, `O(n log n)`, `O(n²)`, or `O(nnz)` only over a stated legal input
domain. Fixed-size, fixed-loop implementations are `Theta(1)` with respect to
inputs they never read, and `Theta(min(n, cap))` as soon as a cap appears in the
code. Studies must derive bounds from actual loops, not from names.

## 2. Source-Link Convention

Every claim about the current implementation links to a tracked symbol:

- **Implementation files:** `source/<kernel>/cuda/<kernel>.cpp`,
  `source/<kernel>/opencl/…`, `source/<kernel>/opencl_cpp/…`.
- **Reference and harness:** `source/<kernel>/main.cpp` plus the CPU function
  inside the implementation file — for example, `sigmoid_cpu` in
  [`source/sigmoid/cuda/sigmoid.cpp`](../../source/sigmoid/cuda/sigmoid.cpp).
- **CMake targets for backend coverage:** the root `CMakeLists.txt` plus each
  `source/<kernel>/CMakeLists.txt`.

Rule (R-4): no equation or bound appears in a study without a source anchor.
Ideal alternatives are cited as design references, not as links implying that
they exist in this repository.

## 3. Cost Model

### 3.1 Core Quantities (R-2)

Each study defines these quantities for one named backend variant:

| Quantity | Meaning | Derived from |
|---|---|---|
| Dimensions | Input shapes and legal domain (which sizes compile and run) | Kernel signatures, launch code, harness |
| CPU work | Arithmetic and passes in the reference routine | Reference loops |
| GPU work `W` | Total primitive arithmetic and memory operations executed across **all** threads and stages; a parallelism-independent total | Sum over innermost loops of all kernels and stages |
| Dependency depth `S` | Critical-path length: the number of steps that cannot overlap because each depends on the previous one | Reduction and scan stage counts, serialized loops |
| Auxiliary space | Shared memory, block-sum buffers, workspace, intermediates | Allocations and `__shared__` declarations |
| Memory traffic | Reads and writes to global memory: the minimum, plus what the implementation actually does (tile loads, re-reads, atomics) | Innermost access pattern |

### 3.2 Finite-Device Model

The ideal lower bound on runtime for a fixed device with `P` execution resources
(SIMT lanes or SM cores) is:

```
T_P  >=  max( W/P , S )
```

`P` is a model of the hardware, not the block grid or the number of launched
threads.

- `W/P`: even perfectly parallel work needs at least `W/P` steps.
- `S`: a dependency chain cannot finish faster than its depth, no matter how
  many resources are idle.

Both terms are lower bounds, so the bound is a `max`, not a sum.

The scheduling estimate `O(W/P + S)` holds only under these assumptions:

1. Work is composable into unit operations schedulable on `P` resources
   (Brent's scheduling assumptions).
2. No stage is left serialized by a lock, atomic, or load imbalance.
3. Compute, not memory, is the binding resource.
4. Launch, synchronization, and transfer overheads are negligible at the sizes
   measured.

A study states these assumptions explicitly. If any of them fails, the estimate
is a hypothesis, not the model.

`P` is never the number of threads launched. Launching 1000× more threads does
not create 1000× more hardware. For fixed `P`, increasing `n` generally keeps
the same asymptotic work growth while adding constant factors (and wave
quantization); it does not change the Big-O in `n`. No study may write an
equation that equates thread count with fixed-device speedup.

### 3.3 Caveats Between the Ideal Bound and Observed Time

Actual time also depends on:

- **Bandwidth.** Achievable GB/s differs from the infinite bandwidth implicitly
  assumed by `W/P`. Bandwidth-bound kernels scale with bytes per element, not
  FLOPs.
- **Host-device transfers.** PCIe traffic is frequently the dominant term for
  small `n` and belongs in the traffic column.
- **Synchronization.** `__syncthreads()` between tile loads, and
  stage-separated kernel launches for multi-pass scans.
- **Front-end constants and launches.** Kernel launch latency and per-block
  scheduling cost `Theta(number of launches)`, independent of `n`. They dominate
  small sizes and floor the curve.
- **Occupancy and resource limits.** Register and shared-memory usage caps
  resident blocks. Wave quantization makes timing step-wise in `n`, and
  occupancy below one wave violates assumption 2 of the Brent estimate.

A study separates these regimes explicitly. A runtime curve in one regime does
not certify the asymptotic bound.

## 4. Improvement Taxonomy (R-3)

Each opportunity in a study is tagged with exactly one primary category, along
with its applicability, numerical consequences, and any lower bound:

1. **Total work.** Reduce `W` — for example, replace a Hillis-Steele
   `O(n log n)` scan with a work-efficient `O(n)` scan, or remove a redundant
   pass. Changing the algorithm's asymptotic work requires a different input
   contract or a fundamentally different scheme; an identical I/O contract
   usually pins `W` from below, because all inputs must be read and all outputs
   written.
2. **Parallel depth.** Reduce `S` without changing `W` — for example, replace a
   serial per-row K loop with a parallel tree reduction over K for GEMM
   accumulators. This improves the `S` term of `T_P` and is useless where `W/P`
   dominates.
3. **Memory space.** Reduce auxiliary storage — for example, avoid
   materializing an intermediate block-sum or score buffer. This overlaps with
   traffic when the extra space is round-tripped through memory.
4. **Memory traffic.** Reduce bytes moved through tiling, shared-memory reuse,
   fusion, or vectorized access. Ordinary GEMM tiling does **not** change the
   `O(MNK)` arithmetic count; it changes reuse, traffic, and constants.
5. **Constant factor.** Keep the asymptotics but shrink the constants:
   `__expf` intrinsics, `float4` vectorization, fewer branches. Always quantify
   where the constant comes from in the source — intrinsic choice, uncoalesced
   stride, or a branchy tail.

Every taxonomy entry also lists its assumptions (shape legality and precision),
numerical consequences (changed accumulation order, fast-math error), the
attainable lower bound, and the cases where **asymptotic improvement is
unavailable** under the fixed dense I/O contract. All proposals stay labeled
**unimplemented / hypothesis** until code, correctness, and measurement exist.

## 5. Correctness Constraints for Future Changes

- Correctness is validated before timing. Harnesses compare against the CPU
  reference within the stated tolerance (`~1e-4f`).
- Reduction re-association (parallel scans, tree reductions) changes
  accumulation order. Tolerance must account for datatype, order, and input
  magnitude.
- Fast-math intrinsics such as `__expf` are approximations. State the precision
  change; do not swap them in silently.
- Odd sizes, `B−1/B/B+1`, one-element, and block-boundary shapes must be legal
  in the harness or explicitly out of contract. Tails from non-multiple-of-4
  vectorized loads deserve an explicit correctness note.
- Zero and invalid sizes are tested only against a defined input contract.
- Source observations logged in the spec — for example, the prefix-sum first
  stage looping to `n` rather than block width, and the single-block block-sum
  stage — are correctness flags to resolve before any optimization.

## 6. Evidence, Hypothesis, and Measurement (R-4, R-6)

Studies label every claim at one of three tiers:

- **Source-derived bound** — evidence about the code, not the hardware. Traced
  from loops, stages, and allocations in linked source.
- **Hypothesis** — a model prediction about hardware behavior, such as
  "bandwidth-bound above n≈…" or "occupancy caps at …". It must state its
  assumptions and how it would be falsified.
- **Measured** — `benchmark()` or `benchmarkWithReset()` medians and spread over
  geometric size sweeps, with environment, timing boundary, and tolerance
  recorded per R-6.

Runtime measurements **support** a model; they never **prove** its Big-O. A
straight line on a log-log plot is consistent with `Theta(n)` over the measured
range only.

## 7. Per-Kernel Study Template

Every study in `docs/complexity/<kernel>.md` uses exactly these sections:

1. **Operation and dimensions** — what it computes, the symbol and shape
   vocabulary, and the legal input domain.
2. **Source links** — implementation symbols, CPU reference, harness, and CMake
   targets for each available backend.
3. **CPU work** — reference passes and their costs.
4. **Actual GPU work and depth** — `W` and `S` for each implemented variant,
   derived stage by stage from the *implemented* algorithm rather than the
   textbook ideal.
5. **Auxiliary storage and traffic** — shared memory, buffers, and
   intermediates; global bytes read and written per call.
6. **Finite-device model** — `T_P >= max(W/P, S)`, the assumptions required to
   use `O(W/P + S)`, and the expected regime (bandwidth, compute, or latency).
7. **Improvement options** — a taxonomy-tagged list per §4, with assumptions,
   numerical consequences, lower bounds, and unavailability notes. All entries
   are labeled unimplemented or hypothesis.
8. **Correctness constraints** — tolerance, reorder effects, shape edge cases,
   and invalid inputs.
9. **Evidence and limitations** — which claims are source-derived,
   hypothetical, or measured, and which hardware or instrumentation is missing.

## 8. Kernel Inventory and Backend Coverage (R-1)

The root CMake defines **19 implemented harness targets**. Backend coverage is
verified against the directory structure and each kernel's `CMakeLists.txt`:

- 15 kernels build with all three backends (`cuda`, `opencl`, `opencl_cpp`).
- `deep_learning_inference` is CUDA-only in practice: it has an `opencl/` source
  directory, but its `CMakeLists.txt` skips OpenCL builds.
- `categorical_cross_entropy`, `fp16_dot_product`, and `gaussian_blur` are
  **CUDA-only**.
- Pending entries are not counted as implemented, and no study may invent
  complexity for them: `lora_linear` has an empty `source/lora_linear/`
  directory with no harness or CMake target; `vector_addition` is Python-only
  under `source/vector_addition/`, with no C++ harness or CMake target.

| Kernel | CUDA | OpenCL | OpenCL C++ | Study | Variants noted in source |
|---|---|---|---|---|---|
| categorical_cross_entropy | yes | — | — | no study yet | CUDA-only |
| convolution2d | yes | yes | yes | no study yet | |
| deep_learning_inference | yes | — | — | no study yet | OpenCL source exists; CMake skips it |
| fp16_dot_product | yes | — | — | no study yet | CUDA-only |
| gaussian_blur | yes | — | — | no study yet | CUDA-only |
| geglu | yes | yes | yes | no study yet | |
| gemm | yes | yes | yes | phase-1 pilot ([study](gemm.md)) | tiled `gemmTiled`, 2×2-register `gemmTiled2x2` |
| interleave_arrays | yes | yes | yes | no study yet | |
| matrix_mul | yes | yes | yes | no study yet | |
| matrix_transpose | yes | yes | yes | no study yet | |
| prefix_sum | yes | yes | yes | phase-1 pilot ([study](prefix-sum.md)) | Hillis-Steele blocked scan (see §9.2) |
| reverse_array | yes | yes | yes | no study yet | |
| rgb_to_grayscale | yes | yes | yes | no study yet | |
| sigmoid | yes | yes | yes | phase-1 pilot ([study](sigmoid.md)) | scalar `sigmoid_kernel`, vectorized `sigmoid_kernel2` |
| silu | yes | yes | yes | no study yet | |
| softmax | yes | yes | yes | no study yet | |
| spmv | yes | yes | yes | no study yet | |
| swiglu | yes | yes | yes | no study yet | |
| value_clipping | yes | yes | yes | no study yet | |
| lora_linear | pending | pending | pending | no study yet | empty directory; no harness or CMake |
| vector_addition | pending | pending | pending | no study yet | Python-only; no C++ harness or CMake |

"Phase-1 pilot" means the study linked in that row, delivered with this phase
per the [plan](../../specs/2026-09-20-kernel-complexity/plan.md). It prestates
no numbers beyond what the linked study derives.

## 9. Pilot Teaching Examples

These three examples anchor the model. They are teaching derivations; the pilot
studies analyze the actual coded variants against them.

### 9.1 Elementwise (Sigmoid): Linear Work Stays Linear

A pointwise function `y[i] = 1/(1+e^(-x[i]))` performs `Theta(1)` work per
element, so `W = Theta(n)` and `S = Theta(1)`: each output depends only on its
input. The ideal bound is `T_P = max(Theta(n/P), Theta(1))`.

**Linear work remains linear.** No restructuring of elementwise code changes
the Big-O. Apart from `n`, the only asymptotic floor is reading `n` inputs and
writing `n` outputs. Vectorization — `float4` loads and stores, four elements
per thread, with `__expf`
([`sigmoid_kernel2`](../../source/sigmoid/cuda/sigmoid.cpp)) — **changes
constants** (fewer memory transactions, cheaper exponentials) and changes
numerics (`__expf` versus `expf` precision), but not the asymptotic class. Its
scalar tail loop preserves correctness at non-multiple-of-4 sizes.

### 9.2 Scan (Prefix Sum): A Work/Depth Trade-off, Derived Rather Than Borrowed

Two ideal schemes compute an inclusive prefix sum over `n` elements:

- **Serial scan:** `W = Theta(n)`, `S = Theta(n)`. There is no parallelism
  without rework.
- **Hillis-Steele** (doubling offsets, `offset *= 2`,
  [`block_inclusive_scan`](../../source/prefix_sum/cuda/prefix_sum.cpp)): each
  of `n` lanes performs `log2 n` steps, so **work `W = Theta(n log n)`** and
  **depth `S = Theta(log n)`**. More total work, far shorter critical path.
- **Work-efficient scan** (Blelloch-style up/down sweep): **`W = Theta(n)`**,
  **`S = Theta(log n)`**. Both `W` and `S` are ideal for this operation. Both
  doubling and work-efficient schemes reach ideal `O(log n)` depth, so the real
  difference is the work term.

The CUDA code is a three-stage blocked scheme: a per-block inclusive scan
(`block_inclusive_scan`, a doubling scan over a block of width `B`), a scan of
the `ceil(n/B)` block sums (`scan_block_sums`), and `add_block_sums`, which
broadcasts each scanned block sum to the block's elements.

Its costs follow from **`B` and `ceil(n/B)`**, and from the loops actually
written: at most `n·log2 B` + `ceil(n/B)·log2(ceil(n/B))` + `n` steps, with a
launch-heavy depth of `log2 B + log2(ceil(n/B)) + 2` kernel stages. These must
be derived from that `B`-dependent structure. Do **not** assign the textbook
`Theta(n log n)` or `Theta(n)` bound to the blocked implementation by analogy:
with finite `B` set by hardware, its stage count, launch count, and auxiliary
`ceil(n/B)` space are properties of this specific code. Hardware limits — `n`
versus block width, and the single-block block-sum stage — remain open
correctness and scaling questions flagged in the spec under "Discovery
evidence".

### 9.3 GEMM: O(MNK) Persists Under Tiling

Classical dense `C[M×N] = A[M×K]·B[K×N]` requires `M·N·K` multiply-adds: every
output element touches every element of its `K`-dimension strip. Thus
**`W = Theta(MNK)`**, and **`S = Theta(K)`** for a serial per-output K loop, in
which one thread accumulates a scalar over K.

Tiling with shared memory ([`gemmTiled`](../../source/gemm/cuda/gemm.cpp), plus
the 2×2 register variant `gemmTiled2x2`) does **not** change this: each output
still consumes K products, so **`O(MNK)` arithmetic persists under ordinary
tiling**. Tiling changes reuse and traffic. Each A or B element is read
`O(N/TILE_K)` or `O(M/TILE_K)` times instead of being re-fetched per output, and
traffic drops from `Theta(MNK·reuse_factor)` toward
`Theta(MNK/max(M,N,TILE)+M·K+K·N)`. Tiling also adds synchronization
(`__syncthreads()` per tile).

A **parallel reduction over K** per output (proposed, unimplemented) would
change `S` from `Theta(K)` to `Theta(log K)` while keeping `W = Theta(MNK)`. It
is a depth-term improvement, not a work-term one, and it adds auxiliary partial
sums and a re-association of the accumulation order.

**The work itself changes only if the input contract supplies structure:**
sparsity (A or B sparse gives `Theta(nnz)` per output strip), low rank (an
effective `K`), diagonal or banded structure, or symmetric batch inputs. Each
can reduce the required multiply-adds below `Theta(MNK)`, but only because the
*new contract* makes whole products guaranteed-zero or shared. A dense in/out
contract requires reads and arithmetic that grow at least proportionally to `MN`
and `MNK` respectively, so neither cost can fall below that floor. The serial-K
versus parallel-K choice affects `T_P` only when the `M·N·P-ratio` supplies more
parallelism than the `M·N` outputs already do; state which regime applies
(W-bound or S-bound) rather than assuming depth wins.

## 10. Coverage Obligations Going Forward

- Phase 2 writes one study per implemented kernel, batched per the
  [plan](../../specs/2026-09-20-kernel-complexity/plan.md). Each study follows
  §7 exactly and links from the README once Phase 3 lands those links.
- Pending entries (`lora_linear`, `vector_addition`) get status notes only.
  Never invent implementation complexity or speedup claims for them.
- Every new kernel added through
  [Adding a new kernel](../guides/adding-a-new-kernel.md) must include a
  complexity study with the change (per R-5).
- Unimplemented ideas stay labeled as hypotheses per §4 and §6 until measured.
  An estimated speedup may never be presented as a measured one.
