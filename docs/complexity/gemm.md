# Kernel Complexity Analysis: GEMM

Phase 1, attempt P1-T3-gemm. Source-derived analysis only — no measurements were
taken or fabricated for this doc; every claim below is traceable to code.

## Operation and Dimensions

GEMM computes `C = alpha * A * B + beta * C`, where `A` is `M x K` row-major,
`B` is `K x N` row-major, `C` is `M x N`
([source/gemm/cuda/gemm.cpp](../../source/gemm/cuda/gemm.cpp),
`gemmCpu` / `gemmTiled`; spec also stated in the file header comment).

Harness pilots in `source/gemm/main.cpp`:

| Pilot | M | K | N | alpha | beta | Purpose |
|-------|---|---|---|-------|------|---------|
| Small (M1/K1/N1) | 2 | 3 | 2 | 1.0 | 0.0 | Hand-checked result `[[58,64],[139,154]]`, boundary/exactness spot check |
| Large (M2/K2/N2) | 512 | 256 | 512 | 1.5 | 0.5 | Performance-meaningful size, 5-seed statistical validation |

Small pilot output (M1*N1=4 elements). Large pilot output M2*N2 = 262144
elements.

## Source Links

- CUDA backend: `source/gemm/cuda/gemm.h`, `source/gemm/cuda/gemm.cpp`
- OpenCL C API backend: `source/gemm/opencl/gemm.h`, `source/gemm/opencl/gemm.cpp`
- OpenCL C++ wrapper backend: `source/gemm/opencl_cpp/gemm.h`,
  `source/gemm/opencl_cpp/gemm.cpp`
- Backend-agnostic driver/harness: `source/gemm/main.cpp`
- CPU reference: `gemmCpu` in the CUDA `gemm.cpp` (opencl backends likewise
  ship dispatch wrappers; the C++-equivalent reference lives in `main.cpp` scope
  via the CUDA variant used to produce comparison numbers).

## CPU Work

`gemmCpu` is a triple loop: for each `(i, j)` a serial K-length dot product is
accumulated, then Epilogue `alpha*sum + beta*C[i][j]`.

- Work: Theta(M*N*K) multiply-accumulate (MAC) operations for the small pilot
  → 2*3*2 = 12 MACs; the large pilot → 512*256*512 = 67,108,864 MACs.
- Scaling additions: Theta(MN) for the epilogue (read + write of C), negligible
  against MNK but not negligible when K ≤ N.
- Space: Theta(MK + KN + MN) for the three matrices; no auxiliary space.

## Actual GPU Work and Depth (Dispatched Kernel)

The dispatched CUDA kernel is `gemmTiled` (TILE_SIZE = 16, dispatched from
`gemmGPU` in `source/gemm/cuda/gemm.cpp`; the OpenCL backends dispatch their
own `gemmTiled` kernel with the same 16x16 tile shape, verified by reading
`source/gemm/opencl/gemm.cpp` and `source/gemm/opencl_cpp/gemm.cpp`).

### Work W

W = Theta(MNK). Tiling and shared memory do NOT change the arithmetic: each of
the MN outputs still accumulates one multiply-add per (i,k) pair, i.e. `TILE_
SIZE`-wide tiles mean each loaded element of A (or B) is reused TILE_SIZE times
instead of evicted between uses, cutting global-memory traffic but not the MAC
count. Work is Theta(MNK) for all three CUDA variants
(`gemmTiled`, `gemmTiled2x2`, `gemmOptimized`).

### Depth S

S = Theta(K) for the dispatched kernel. Each thread serially accumulates its
dot product along K: the loop
`for (t = 0; t < numTiles; t++) { for (k = 0; k < TILE_SIZE; k++) sum += ... ; }`
chains `numTiles * TILE_SIZE ~= K` dependent FMA steps into a single register
accumulator. Splitting K across threads (parallel reduction) is **not
implemented** here and is labeled an unimplemented proposal; such a design
would reach depth Theta(log K) for the in-tile step (tree reduce with the same
Theta(MNK) total work) but needs extra reduction space and an extra
synchronization round.

### Grid Shape

For the large pilot (M=N=512, TILE_SIZE=16): a grid of
(512/16) x (512/16) = 32 x 32 = 1024 blocks, each block 16x16 = 256 threads,
one output element per thread. Small pilot (M=2, N=2) still allocates one
16x16 block, with the same guarded `row < M && col < N` write path (loads are
also guarded; out-of-range tiles read as 0).

## Auxiliary Storage and Traffic

Keywords: auxiliary space; memory traffic; finite-resource model.

Per-block auxiliary storage (dispatched `gemmTiled`): two 16x16 float shared
arrays, `As[TILE_SIZE][TILE_SIZE]` and `Bs[TILE_SIZE][TILE_SIZE]` — 2 x 256
x 4 B = 2 KiB per block (CUDA and both OpenCL variants use the identical
shape).

Reuse factor: each element of A brought into shared memory is consumed
TILE_SIZE = 16 times inside the block (same for B) before the next K-tile
replaces it. Higher reuse under `gemmTiled2x2` (2 outputs per thread share
loaded A and B values) is **not dispatched** — unimplemented alternative.

Global memory traffic models (arithmetic totals, ignoring cache effects —
L2/L1 can reduce the naive figures and are not claimed here):

- Naive per-thread streaming (no tiling): Theta(MK) reads of A + Theta(KN)
  reads of B per output row/column → Theta(MNK + MNK) if every output reloads
  its K-slice, i.e. approx Theta(MN(M+N)) when K-slices are re-fetched per
  output; the tiling model in `gemmTiled` is Theta(MNK / TILE_SIZE) global
  loads plus Theta(MN) for the final C read + write.
- C read + write both required because of `beta * C`: the kernel performs
  `c_val = C[row*N + col]` and then a full `C[row*N + col] = result` even when
  beta = 0; both directions are counted once per output element.

Although the beta*C reuse factor means large-pilot traffic is dominated by MAC
count once per (i,k) pair from shared memory, adding a proper
epilogue-fusion/`beta`-fast-path is out of scope for Phase 1 and is NOT claimed
to be implemented.

## Finite-Device Model

Keywords: auxiliary space; memory traffic; finite-resource model.

- `T_P >= max(W/P, S)` with `W` = Theta(MNK) and `S` = Theta(K) for the
  dispatched kernel; the roofline lower bound is the larger of the two.
- `P` is a hardware resource model (SM count, max resident threads/registers/
  shared memory per SM), NOT a thread count. In this phase the analytical
  model is derived without profiling: no occupancy, warp-scheduler behavior,
  bank-conflict, or coalescing measurements back the per-SM residency claims.
- Occupancy / bank-conflict / coalescing caveats: `gemmOptimized` (with
  `+1`-padded shared arrays to avoid bank conflicts, i.e.
  `As[TILE_SIZE][TILE_SIZE + 1]`, and explicit `fmaf`) is present in
  `source/gemm/cuda/gemm.cpp` but NOT dispatched by `gemmGPU`. Bank-conflict
  avoidance and row-major coalescing in that kernel are labeled unimplemented
  alternatives for the purposes of the dispatched-path analysis. The dispatched
  `gemmTiled` uses unpadded 16x16 shared tiles (potential bank conflicts on
  `Bs[k][tx]` reads) and row-major loads along `tx`.
- Launch caveats: per-call costs (cudaMalloc / cudaMemcpy H2D and D2H /
  kernel launch) sit outside the kernel but inside the harness window; for the
  small pilot these dominate any GPU kernel work.
- Fixed-P scaling: with P fixed, `W/P = Theta(MNK)` retains the MNK growth
  rate; asymptotic scaling is unchanged by any constant-factor tiling change —
  only P (more SMs, more resident threads) or a sub-MNK algorithm would move it.
- The `W/P` model treats the small pilot's 12-MAC problem as a degenerate case:
  grid there is 1 block, so the roofline is effectively serial, not one block
  per 9 SMs.

## Improvement Options (Tagged)

| Option | Category / what it promises | Status here |
|--------|----------------------------|-------------|
| 2x2 thread tiling / register blocking (`gemmTiled2x2`, TILE_M=32, TILE_N=32, TILE_K=16, THREAD_TILE=2) | constant-factor arithmetic-intensity and traffic improvement; W stays Theta(MNK) | Present in `source/gemm/cuda/gemm.cpp` but NOT dispatched from `gemmGPU` — unimplemented alternative, not claimed to help measured numbers |
| Parallel-K split + cross-thread/tree reduction | parallel-depth improvement (S from Theta(K) toward Theta(log K) or Theta(K/P)); costs extra space and an extra reduction sync; W stays Theta(MNK) | NOT implemented; hypothesized proposal only |
| Tensor Cores / cuBLAS | constant-factor speedup with precision/range and layout constraints (e.g. FP16/TF32 inputs, alignment) | NOT implemented, not claimed |
| Sparsity / Strassen | sub-MNK total-work reduction | NOT claimed — only valid if the input contract supplies structure (zeros, block sparsity, or approximate-block decomposition), which this harness does not |
| `+1` shared-memory padding + FMA (`gemmOptimized`) | constant-factor memory-pipeline and instruction-mix improvement | Present but NOT dispatched — unimplemented alternative |

## Correctness Constraints

- FP32 accumulation order: the CPU reference is a strict `i, j, k` serial
  accumulation; the dispatched CUDA/OpenCL kernel accumulates in
  tile-major (`t` outer, `k` inner) order, i.e. a different FP32 rounding order
  than the CPU. Differences are bounded by tolerance and are validated, not
  eliminated.
- Tolerance: `compareResults` in `source/gemm/main.cpp` uses an absolute
  tolerance of `0.1f`, which is loose relative to typical FP32 GEMM error for
  these magnitudes and inputs; correctness claims should state the tolerance
  used rather than "exact".
- `beta * C` semantics: `C` is both input and output, so repeated calls
  accumulate into their own result. `main.cpp` uses `benchmarkWithReset(...)`
  with a `C_seed` copy so every timed iteration starts from the same C
  (explicitly noted in a comment in `source/gemm/main.cpp`).
- 5-seed validation: random data is generated for seeds 42, 1337, 2024, 7, 99
  over the large pilot; a single random draw was judged insufficient to expose
  boundary behaviors such as non-multiple-of-16 M, N, K tile edges (although
  the pilots here are all multiples of 16, the guards `row < M`, `aCol < K`,
  etc. exist for the general case).

## Evidence and Limitations

- This doc is source-derived only: no speedup, bandwidth, or occupancy was
  measured or claimed. Numerical performance statements that would require
  benchmark runs are avoided or explicitly labeled as unmeasured.
- The OpenCL (C API and C++ wrapper) backends were read to confirm they
  dispatch a `gemmTiled` kernel with the same 16x16 tile and a 2D block shape
  matching `TILE_SIZE`, but their runtime behavior (build success, correctness
  on this machine, achieved throughput) was not exercised during this attempt.
- `gemmTiled2x2` and `gemmOptimized` are labeled as unimplemented alternatives
  for this analysis because the dispatched `gemmGPU` path selects `gemmTiled`
  exclusively in the CUDA backend.
- Any Phase 1 conclusions about GEMM's roofline on real hardware must come
  from a later profiling attempt, not from this doc.
