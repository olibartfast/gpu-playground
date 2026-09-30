# Prefix Sum (Inclusive Scan) — Complexity Study (Phase 1, P1-T3-prefix)

Status: source-derived documentation only. No builds, no timing, no executed
failure reproduction were performed in this packet.

## 1. Operation and Dimensions

Inclusive prefix scan (prefix sum) over `n` floats:

```
y[i] = sum_{j=0..i} x[j]
```

Dimension space: a single axis `n`. Example in the harness: `n = 4` with input
`[1, 2, 3, 4]` and expected output `[1, 3, 6, 10]`
(`source/prefix_sum/main.cpp`).

## 2. Source Links

CUDA backend (primary object of study):

- `source/prefix_sum/cuda/prefix_sum.cpp`
- `source/prefix_sum/cuda/prefix_sum.h`
- `source/prefix_sum/main.cpp` (N=4 harness, `benchmark()` from
  `source/utils/benchmark.h`, expected `[1, 3, 6, 10]`, tolerance `1e-5`)

Other backends (not read in this packet; parity to verify in a later task):

- `source/prefix_sum/opencl/prefix_sum.cpp`, `prefix_sum.h`
- `source/prefix_sum/opencl_cpp/prefix_sum.cpp`, `prefix_sum.h`

Specs consulted: `specs/2026-09-20-kernel-complexity/requirements.md`,
`plan.md`, `validation.md`.

## 3. CPU Work

`prefix_scan_cpu` is a serial running sum:

```
W_cpu = Theta(n),  D_cpu = Theta(n)  (strict serial dependence)
```

## 4. GPU Work and Depth — Derived From the Implementation

Config constants from `prefix_scan_gpu`: block width `B = 256` threads,
`m = ceil(n / B)` blocks. A single-block fast path is used when `n <= B`
(kernel 1 launched once with a **null** `block_sums` pointer); otherwise the
three-stage path runs.

Stage 1 — `block_inclusive_scan` (Hillis–Steele per block, shared memory):

- As coded, the scan loop is `for (offset = 1; offset < n; offset *= 2)`,
  i.e. it runs `ceil(log2 n)` iterations bounded by **n, not by the block
  width B** (`prefix_sum.cpp:28`). Iterations past `log2 B` do redundant
  `temp[tid - offset]` reads / guarded adds and two extra `__syncthreads()`
  each. Source observation, recorded as an open correctness/perf item in
  Section 8; not executed in this packet.
- Work for the whole device, as-coded: every one of `m` blocks performs
  `B * ceil(log2 n)` read/add steps → `W1 = Theta(B * m * log n) = Theta(n log n)`
  (since `B * m = Theta(n)`).
- Ideal-coding version (loop to `B`): `W1_ideal = Theta(n log B) = Theta(n)`
  for fixed B, depth `O(log B)`.
- Depth: `D1 = O(log n)` as coded (ideal `O(log B)`), plus two
  `__syncthreads()` barriers per iteration.

Stage 2 — `scan_block_sums` (single block Hillis–Steele over the `m`
partial sums):

- Launched as `<<<1, m>>>` with `m` shared floats.
- Work as coded: `W2 = Theta(m log m)`; depth `D2 = O(log m)`. A
  work-efficient single-block scan would be `W2 = O(m)`.
- When `n >= B + 1`, this invoke adds one launch, one H2D-independent
  global read (m floats) and one global write (m floats).

Stage 3 — `add_block_sums` (elementwise add of each block's predecessor
total):

- One thread per element across `m` blocks:
  `W3 = Theta(n)`, depth `D3 = O(1)`.

Stage totals (as coded, multiblock path):

| Stage                    | Work as coded        | Ideal work | Ideal depth |
|--------------------------|----------------------|------------|-------------|
| 1: per-block scan        | Theta(n log n)       | Theta(n log B) = Theta(n) | O(log B) |
| 2: scan of m block sums  | Theta(m log m)       | O(m)       | O(log m)    |
| 3: add back              | Theta(n)             | Theta(n)   | O(1)        |
| **Total**                | **Theta(n log n)** — superlinear | Theta(n) | O(log B + log m) + 3 launches |

Ideal total depth: `O(log n)` overall, but serialized across **3 kernel
launches** in the multiblock path (plus, inside each scan stage, two
`__syncthreads()` per doubling iteration).

## 5. Ideal Alternatives (Textbook, Labeled Unimplemented)

- **Global Hillis–Steele** (all `n` elements, one block/grid-wide shuffle
  style doubling): work `O(n log n)`, depth `O(log n)`. Not implemented in
  this repo's prefix_sum; the blocked scheme above only approximates it.
- **Work-efficient Blelloch scan** (reduce then downsweep): work `O(n)`,
  depth `O(log n)`. The asymptotically minimal choice for large `n`. Not
  implemented in this repo's prefix_sum.

The current implementation is therefore superlinear in work versus
Blelloch (`Theta(n log n)` as coded) while only reaching the same ideal
depth class (`O(log n)`) before multi-launch serialization overheads.

## 6. Auxiliary Storage and Traffic

Keywords: auxiliary space; memory traffic; finite-resource model.

Per the CUDA host path (`prefix_scan_gpu`):

- Shared memory: `B` floats per block (`threadsPerBlock * sizeof(float)` =
  1 KiB) for kernel 1; `m` floats for kernel 2 (only in the multiblock
  path).
- Global auxiliary: `d_block_sums` of `m = ceil(n/B)` floats, allocated in
  the multiblock path only (`prefix_sum.cpp:103`). **Not allocated on the
  `N <= threadsPerBlock` fast path**, where a null pointer is passed to
  `block_inclusive_scan` (see Section 8).
- Global traffic: H2D input copy (n floats), one full global read + write
  per scan stage (kernel 1: n floats each; kernel 2: m floats each),
  kernel 3 adds a global read/write of the output plus the read of
  `d_block_sums`, then D2H output copy (n floats). Total global traffic is
  `O(n)` with a constant larger than the two unavoidable H2D/D2H copies.
- Temporaries are freed at the end of the call; the routine is
  out-of-place (`input`, `output`, `block_sums` are distinct buffers).

## 7. Finite-Device Model

Keywords: auxiliary space; memory traffic; finite-resource model.

- Per the plan (`specs/2026-09-20-kernel-complexity/plan.md`)'s implied
  span/principle: with `P` processors,
  `T_P >= max(W/P, S)` where `W` is total work and `S` span (depth).
- Work `W = Theta(n log n)` (as coded) bounds `T_P/P`; span
  `S = O(log n)` per stage plus launch/sync overhead bounds the critical
  path, so the single-block stage-2 (`scan_block_sums`) is a serialization
  bottleneck: all `m` partial sums are scanned by one block with
  `min(m, B')` effective parallelism, `B'` capped by the device max
  threads-per-block (the launch `<<<1, m>>>` itself does not clamp `m`).
- Occupancy caveats: kernel 1 consumes 1 KiB shared memory per block
  (small, but the loop-to-n bound inflates barrier counts, hurting
  latency-bound progress); kernel 2 is single-block and quotes no
  occupancy benefit.
- `__syncthreads()` cost: two barriers per doubling iteration in the two
  scan kernels — `2 * ceil(log2 n)` and `2 * ceil(log2 m)` barriers per
  block respectively, as coded.
- Launch overhead: 1 launch (fast path) or 3 launches (multiblock path).
- Hardware limits (max threads/block, shared memory, grid dims) must be
  inspected before generalizing these counts to arbitrary `n` — per
  `requirements.md:44-47`, "Inspect hardware limits before generalizing."

## 8. Correctness Constraints and Open Correctness Tasks

- **Float accumulation order**: Hillis–Steele reassociates the sum; GPU
  results are not bit-identical to the serial CPU running sum. A tolerance
  must reflect magnitude (relative error over `sum |x_j|`), not a fixed
  `1e-5` absolute check as in the current harness.
- **Harness coverage**: `main.cpp` validates only `n = 4` against
  `[1, 3, 6, 10]`. Inputs of size 1, odd sizes, `B-1 / B / B+1` (255/256/257)
  and multiblock boundaries are untested (see `validation.md:18`).
- **N=256 null-pointer observation**: on the `n <= threadsPerBlock` fast
  path, `block_inclusive_scan` is passed `nullptr` for `block_sums`
  (`prefix_sum.cpp:100`). The guarded store at `prefix_sum.cpp:41-43`
  writes only when the block's last element index `< n`, i.e. writes
  `temp[255]` to `block_sums[0]` whenever `n == 256` exactly — which
  dereferences the null pointer at `n == 256`. Source observation,
  recorded as a candidate defect for a **separate correctness task**
  (`requirements.md:47`). NOT executed or reproduced in this packet.
- **Block-width loop observation**: the stage-1 loop bound `offset < n`
  instead of `offset < block_size` (`prefix_sum.cpp:28`) is the second
  source-level defect noted in `requirements.md:44-45`; it makes the
  guarded-update condition `tid < n` trivially true for the fast path and
  makes block threads perform `log2 n` doubling rounds over a `B`-lane
  shared buffer. Same disposition: a separate correctness/perf task, not
  executed here.

## 9. Evidence and Limitations

- All work/depth/traffic figures above are **derived from reading the CUDA
  source** (`source/prefix_sum/cuda/prefix_sum.cpp`, `main.cpp`) and the
  spec notes in `specs/2026-09-20-kernel-complexity/requirements.md`.
- No builds were run and **no timing is claimed** in this packet. Any
  performance comparison (`printBench`, speedups in `main.cpp`) is out of
  scope and unexecuted.
- The OpenCL and OpenCL C++ backends were **listed only** (`glob`), not
  read; backend parity with the CUDA three-stage scheme must be verified
  before any of the complexity claims above are attributed to those
  backends.
- The ideal alternatives in Section 5 (global Hillis–Steele, Blelloch
  work-efficient scan) are labeled **unimplemented**; they are contrast
  points for improvement hypotheses, not present code.

## 10. Improvement Options (Hypotheses, Tagged by Category)

| Option | Category | Hypothesis |
|--------|----------|------------|
| Replace blocked Hillis–Steele with a work-efficient (Blelloch reduce/downsweep) scan | total-work improvement | lowers `W` from Theta(n log n) to Theta(n); `T_P >= W/P` bound drops accordingly |
| Fold the loop bound from `n` to `block_size` (B) in kernel 1 | total-work improvement + correctness | removes redundant rounds and barrier traffic; precondition for a clean Theta(n log B) cost model |
| Single-pass / decoupled look-back scan | depth/launch improvement | removes the 3-launch serialization and the single-block stage-2 bottleneck; one kernel, adaptive spin-wait on predecessor flags |
| Vectorized loads (e.g. float4) in kernel 1 global reads/writes | constant factor | fewer memory transactions per element, no algorithmic change |
| In-place vs out-of-place (scan into `input` or fold add-back into a fused pass) | memory-space improvement | reduces auxiliary global buffers/traffic (`d_block_sums` reuse or elimination) |

Each option needs its own single-change packet with correctness validation
before any timing claim, per repo rules.
