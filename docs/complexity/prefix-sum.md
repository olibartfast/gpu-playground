# Prefix Sum (Inclusive Scan): Complexity Study

> Audience: humans, agents · Applies to: CUDA, OpenCL · Last reviewed: 2026-10

Phase 1, attempt `P1-T3-prefix`. This is source-derived documentation only: no
builds, timing, or executed failure reproduction were performed for this study.

## 1. Operation and Dimensions

Inclusive prefix scan (prefix sum) over `n` floats:

```
y[i] = sum_{j=0..i} x[j]
```

The dimension space is a single axis, `n`. The harness uses `n = 4` with input
`[1, 2, 3, 4]` and expected output `[1, 3, 6, 10]`
(`source/prefix_sum/main.cpp`).

## 2. Source Links

CUDA backend, the primary object of this study:

- `source/prefix_sum/cuda/prefix_sum.cpp`
- `source/prefix_sum/cuda/prefix_sum.h`
- `source/prefix_sum/main.cpp` — the `N=4` harness, using `benchmark()` from
  `source/utils/benchmark.h`, expecting `[1, 3, 6, 10]` within tolerance `1e-5`

Other backends were not read for this study; parity remains to be verified in a
later task:

- `source/prefix_sum/opencl/prefix_sum.cpp`, `prefix_sum.h`
- `source/prefix_sum/opencl_cpp/prefix_sum.cpp`, `prefix_sum.h`

Specifications consulted: `specs/2026-09-20-kernel-complexity/requirements.md`,
`plan.md`, and `validation.md`.

## 3. CPU Work

`prefix_scan_cpu` is a serial running sum:

```
W_cpu = Theta(n),  D_cpu = Theta(n)  (strict serial dependence)
```

## 4. GPU Work and Depth, Derived From the Implementation

From `prefix_scan_gpu`: block width `B = 256` threads, `m = ceil(n / B)` blocks.
A single-block fast path runs when `n <= B`, with kernel 1 launched once and a
**null** `block_sums` pointer. Otherwise the three-stage path runs.

### Stage 1: `block_inclusive_scan`

This is a per-block Hillis-Steele scan in shared memory.

- As coded, the scan loop is `for (offset = 1; offset < n; offset *= 2)`, so it
  runs `ceil(log2 n)` iterations bounded by **`n`, not by the block width `B`**
  (`prefix_sum.cpp:28`). Iterations past `log2 B` perform redundant
  `temp[tid - offset]` reads and guarded adds, plus two extra
  `__syncthreads()` each. This source observation is recorded as an open
  correctness and performance item in §8; it was not executed in this study.
- Work for the whole device, as coded: every one of `m` blocks performs
  `B * ceil(log2 n)` read/add steps, so
  `W1 = Theta(B * m * log n) = Theta(n log n)` (because `B * m = Theta(n)`).
- With the ideal loop bound of `B`: `W1_ideal = Theta(n log B) = Theta(n)` for
  fixed `B`, with depth `O(log B)`.
- Depth: `D1 = O(log n)` as coded (ideal `O(log B)`), plus two
  `__syncthreads()` barriers per iteration.

### Stage 2: `scan_block_sums`

This is a single-block Hillis-Steele scan over the `m` partial sums.

- Launched as `<<<1, m>>>` with `m` shared floats.
- Work as coded: `W2 = Theta(m log m)`; depth `D2 = O(log m)`. A work-efficient
  single-block scan would use `W2 = O(m)`.
- When `n >= B + 1`, this stage adds one launch, one global read of `m` floats,
  and one global write of `m` floats.

### Stage 3: `add_block_sums`

This stage adds each block's predecessor total to every element in the block.

- One thread per element across `m` blocks:
  `W3 = Theta(n)`, depth `D3 = O(1)`.

Stage totals for the multiblock path:

| Stage | Work as coded | Ideal work | Ideal depth |
|-------|---------------|------------|-------------|
| 1: per-block scan | `Theta(n log n)` | `Theta(n log B) = Theta(n)` | `O(log B)` |
| 2: scan of `m` block sums | `Theta(m log m)` | `O(m)` | `O(log m)` |
| 3: add back | `Theta(n)` | `Theta(n)` | `O(1)` |
| **Total** | **`Theta(n log n)` — superlinear** | `Theta(n)` | `O(log B + log m)` plus 3 launches |

The ideal total depth is `O(log n)`, but it is serialized across **three kernel
launches** in the multiblock path. Inside each scan stage, there are also two
`__syncthreads()` barriers per doubling iteration.

## 5. Ideal Alternatives (Textbook, Unimplemented)

- **Global Hillis-Steele** over all `n` elements in a single block or grid-wide
  shuffle-style doubling: work `O(n log n)`, depth `O(log n)`. It is not
  implemented in this repository's `prefix_sum`; the blocked scheme above only
  approximates it.
- **Work-efficient Blelloch scan** (reduce, then downsweep): work `O(n)`, depth
  `O(log n)`. This is the asymptotically minimal choice for large `n`, and it is
  not implemented in this repository's `prefix_sum`.

The current implementation is therefore superlinear in work compared with
Blelloch (`Theta(n log n)` as coded), while reaching the same ideal depth class
(`O(log n)`) before multi-launch serialization overheads.

## 6. Auxiliary Storage and Traffic

Keywords: auxiliary space; memory traffic; finite-resource model.

From the CUDA host path (`prefix_scan_gpu`):

- **Shared memory:** `B` floats per block (`threadsPerBlock * sizeof(float)`, or
  1 KiB) for kernel 1; `m` floats for kernel 2, only in the multiblock path.
- **Global auxiliary:** `d_block_sums`, holding `m = ceil(n/B)` floats, is
  allocated only in the multiblock path (`prefix_sum.cpp:103`). It is **not
  allocated on the `N <= threadsPerBlock` fast path**, where a null pointer is
  passed to `block_inclusive_scan` (see §8).
- **Global traffic:** an H2D copy of the input (`n` floats); one full global
  read and write per scan stage (kernel 1: `n` floats each; kernel 2: `m` floats
  each); then kernel 3 reads and writes the output plus reads `d_block_sums`;
  finally a D2H copy of the output (`n` floats). Total global traffic is `O(n)`,
  with a constant larger than the two unavoidable H2D/D2H copies.
- Temporaries are freed at the end of the call. The routine is out-of-place:
  `input`, `output`, and `block_sums` are distinct buffers.

## 7. Finite-Device Model

Keywords: auxiliary space; memory traffic; finite-resource model.

- Following the span principle implied by the plan
  (`specs/2026-09-20-kernel-complexity/plan.md`), with `P` processors:
  `T_P >= max(W/P, S)`, where `W` is total work and `S` is span (depth).
- Work `W = Theta(n log n)` as coded bounds `T_P * P`; span `S = O(log n)` per
  stage, plus launch and synchronization overhead, bounds the critical path. The
  single-block stage 2 (`scan_block_sums`) is therefore a serialization
  bottleneck: all `m` partial sums are scanned by one block with `min(m, B')`
  effective parallelism, where `B'` is capped by the device's maximum
  threads-per-block. The launch `<<<1, m>>>` itself does not clamp `m`.
- **Occupancy caveats:** kernel 1 consumes 1 KiB of shared memory per block.
  That is small, but the loop-to-`n` bound inflates barrier counts and hurts
  latency-bound progress. Kernel 2 is single-block and claims no occupancy
  benefit.
- **`__syncthreads()` cost:** two barriers per doubling iteration in the two
  scan kernels, or `2 * ceil(log2 n)` and `2 * ceil(log2 m)` barriers per block
  respectively, as coded.
- **Launch overhead:** 1 launch on the fast path, 3 on the multiblock path.
- Hardware limits (maximum threads per block, shared memory, grid dimensions)
  must be inspected before generalizing these counts to arbitrary `n`. Per
  `requirements.md:44-47`: "Inspect hardware limits before generalizing."

## 8. Correctness Constraints and Open Tasks

- **Float accumulation order:** Hillis-Steele reassociates the sum, so GPU
  results are not bit-identical to the serial CPU running sum. Tolerance must
  reflect magnitude (relative error over `sum |x_j|`), not the fixed `1e-5`
  absolute check used by the current harness.
- **Harness coverage:** `main.cpp` validates only `n = 4` against
  `[1, 3, 6, 10]`. Inputs of size 1, odd sizes, `B-1 / B / B+1` (255/256/257),
  and multiblock boundaries are untested (see `validation.md:18`).
- **`N=256` null-pointer observation:** on the `n <= threadsPerBlock` fast path,
  `block_inclusive_scan` receives `nullptr` for `block_sums`
  (`prefix_sum.cpp:100`). The guarded store at `prefix_sum.cpp:41-43` writes
  only when the block's last element index is `< n`, so it writes `temp[255]` to
  `block_sums[0]` whenever `n == 256` exactly — dereferencing the null pointer.
  This is a source observation and a candidate defect for a **separate
  correctness task** (`requirements.md:47`); it was not executed or reproduced
  in this study.
- **Block-width loop observation:** the stage-1 loop bound `offset < n` instead
  of `offset < block_size` (`prefix_sum.cpp:28`) is the second source-level
  defect noted in `requirements.md:44-45`. It makes the guarded update condition
  `tid < n` trivially true on the fast path and makes block threads perform
  `log2 n` doubling rounds over a `B`-lane shared buffer. This has the same
  disposition: a separate correctness and performance task, not executed here.

## 9. Evidence and Limitations

- All work, depth, and traffic figures above are **derived from reading the CUDA
  source** (`source/prefix_sum/cuda/prefix_sum.cpp` and `main.cpp`) and the spec
  notes in `specs/2026-09-20-kernel-complexity/requirements.md`.
- No builds were run, and **no timing is claimed**. Any performance comparison
  (`printBench`, speedups in `main.cpp`) is out of scope and unexecuted.
- The OpenCL and OpenCL C++ backends were **listed only**, not read. Backend
  parity with the CUDA three-stage scheme must be verified before attributing
  any of these complexity claims to those backends.
- The ideal alternatives in §5 (global Hillis-Steele and the Blelloch
  work-efficient scan) are labeled **unimplemented**. They are contrast points
  for improvement hypotheses, not present code.

## 10. Improvement Options (Hypotheses, Tagged by Category)

| Option | Category | Hypothesis |
|--------|----------|------------|
| Replace the blocked Hillis-Steele scan with a work-efficient Blelloch reduce/downsweep scan | Total-work improvement | Lowers `W` from `Theta(n log n)` to `Theta(n)`, so the `T_P >= W/P` bound drops accordingly |
| Fold the loop bound from `n` to `block_size` (`B`) in kernel 1 | Total-work improvement and correctness | Removes redundant rounds and barrier traffic; a precondition for a clean `Theta(n log B)` cost model |
| Single-pass or decoupled look-back scan | Depth and launch improvement | Removes the 3-launch serialization and the single-block stage-2 bottleneck; one kernel with an adaptive spin-wait on predecessor flags |
| Vectorized loads (for example `float4`) in kernel 1 global reads and writes | Constant factor | Fewer memory transactions per element, with no algorithmic change |
| In-place rather than out-of-place scan, or fold the add-back into a fused pass | Memory-space improvement | Reduces auxiliary global buffers and traffic (`d_block_sums` reuse or elimination) |

Each option needs its own single-change packet with correctness validation
before any timing claim, per repository rules.
