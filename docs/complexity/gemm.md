# GEMM: Complexity Study

> Audience: humans, agents · Applies to: CUDA, OpenCL · Last reviewed: 2026-10

Phase 1, attempt `P1-T3-gemm`. This is a source-derived analysis: no
measurements were taken or fabricated, and every claim below is traceable to
code.

## Operation and Dimensions

GEMM computes `C = alpha * A * B + beta * C`, where `A` is `M x K` row-major,
`B` is `K x N` row-major, and `C` is `M x N`. The implementation is in
[`source/gemm/cuda/gemm.cpp`](../../source/gemm/cuda/gemm.cpp) (`gemmCpu` and
`gemmTiled`); the file header comment states the same contract.

The harness in `source/gemm/main.cpp` defines two pilots:

| Pilot | M | K | N | alpha | beta | Purpose |
|-------|---|---|---|-------|------|---------|
| Small (M1/K1/N1) | 2 | 3 | 2 | 1.0 | 0.0 | Hand-checked result `[[58,64],[139,154]]`; boundary and exactness spot check |
| Large (M2/K2/N2) | 512 | 256 | 512 | 1.5 | 0.5 | Performance-meaningful size; 5-seed statistical validation |

The small pilot produces `M1*N1 = 4` output elements; the large pilot produces
`M2*N2 = 262144`.

## Source Links

- **CUDA backend:** `source/gemm/cuda/gemm.h`, `source/gemm/cuda/gemm.cpp`
- **OpenCL C API backend:** `source/gemm/opencl/gemm.h`,
  `source/gemm/opencl/gemm.cpp`
- **OpenCL C++ wrapper backend:** `source/gemm/opencl_cpp/gemm.h`,
  `source/gemm/opencl_cpp/gemm.cpp`
- **Backend-agnostic driver and harness:** `source/gemm/main.cpp`
- **CPU reference:** `gemmCpu` in the CUDA `gemm.cpp`. The OpenCL backends ship
  equivalent dispatch wrappers; the comparisons here use the CUDA variant.

## CPU Work

`gemmCpu` is a triple loop: for each `(i, j)`, it accumulates a serial K-length
dot product, then applies the epilogue `alpha*sum + beta*C[i][j]`.

- **Work:** `Theta(M*N*K)` multiply-accumulate (MAC) operations. The small pilot
  performs `2*3*2 = 12` MACs; the large pilot performs
  `512*256*512 = 67,108,864`.
- **Scaling additions:** `Theta(MN)` for the epilogue (read and write of `C`).
  This is negligible against `MNK`, but not negligible when `K <= N`.
- **Space:** `Theta(MK + KN + MN)` for the three matrices, with no auxiliary
  space.

## Actual GPU Work and Depth (Dispatched Kernel)

The dispatched CUDA kernel is `gemmTiled`, with `TILE_SIZE = 16`, launched from
`gemmGPU` in `source/gemm/cuda/gemm.cpp`. The OpenCL backends dispatch their own
`gemmTiled` kernel with the same 16x16 tile shape, verified by reading
`source/gemm/opencl/gemm.cpp` and `source/gemm/opencl_cpp/gemm.cpp`.

### Work W

`W = Theta(MNK)`. Tiling and shared memory do **not** change the arithmetic:
each of the `MN` outputs still accumulates one multiply-add per `(i, k)` pair.
A `TILE_SIZE`-wide tile means each loaded element of A or B is reused
`TILE_SIZE` times instead of being evicted between uses, which cuts global-memory
traffic but not the MAC count. Work is `Theta(MNK)` for all three CUDA variants:
`gemmTiled`, `gemmTiled2x2`, and `gemmOptimized`.

### Depth S

`S = Theta(K)` for the dispatched kernel. Each thread serially accumulates its
dot product along K:

```cpp
for (t = 0; t < numTiles; t++) {
    for (k = 0; k < TILE_SIZE; k++) sum += ...;
}
```

This chains `numTiles * TILE_SIZE ~= K` dependent FMA steps into a single
register accumulator. Splitting K across threads (a parallel reduction) is **not
implemented** and is labeled an unimplemented proposal. Such a design would
reach depth `Theta(log K)` for the in-tile step, with the same `Theta(MNK)` total
work, but it needs extra reduction space and an extra synchronization round.

### Grid Shape

For the large pilot (`M=N=512`, `TILE_SIZE=16`), the grid is
`(512/16) x (512/16) = 32 x 32 = 1024` blocks, each with `16x16 = 256` threads
and one output element per thread. The small pilot (`M=2`, `N=2`) still
allocates one 16x16 block and uses the same guarded `row < M && col < N` write
path. Loads are also guarded, and out-of-range tiles read as zero.

## Auxiliary Storage and Traffic

Keywords: auxiliary space; memory traffic; finite-resource model.

The dispatched `gemmTiled` uses two 16x16 float shared arrays per block,
`As[TILE_SIZE][TILE_SIZE]` and `Bs[TILE_SIZE][TILE_SIZE]`: `2 x 256 x 4 B = 2
KiB` per block. The CUDA and both OpenCL variants use the identical shape.

The reuse factor is `TILE_SIZE = 16`: each element of A brought into shared
memory is consumed 16 times inside the block before the next K-tile replaces
it; the same holds for B. `gemmTiled2x2` has higher reuse, because two outputs
per thread share loaded A and B values, but it is **not dispatched** and is
therefore an unimplemented alternative.

Global-memory traffic models below are arithmetic totals that ignore cache
effects. L2 and L1 can reduce the naive figures, and such reductions are not
claimed here.

- **Naive per-thread streaming (no tiling):** `Theta(MK)` reads of A plus
  `Theta(KN)` reads of B per output row and column. If every output reloads its
  K-slice, this becomes `Theta(MNK + MNK)`, or approximately
  `Theta(MN(M+N))` when K-slices are re-fetched per output.
- **Tiled `gemmTiled`:** `Theta(MNK / TILE_SIZE)` global loads plus `Theta(MN)`
  for the final read and write of C.
- **`beta * C`:** the kernel reads `C[row*N + col]` and writes
  `C[row*N + col]` even when `beta = 0`, so both directions count once per
  output element.

Because every output is both a read and a write, the large pilot's traffic
includes that epilogue cost on top of the MAC-driven shared-memory traffic. A
dedicated epilogue-fusion or beta fast path is out of scope for Phase 1 and is
not claimed to be implemented.

## Finite-Device Model

Keywords: auxiliary space; memory traffic; finite-resource model.

- `T_P >= max(W/P, S)`, with `W = Theta(MNK)` and `S = Theta(K)` for the
  dispatched kernel. The roofline lower bound is the larger of the two terms.
- `P` is a hardware resource model (SM count, maximum resident
  threads/registers/shared memory per SM), **not** a thread count. This phase
  derives the analytical model without profiling: no occupancy,
  warp-scheduler, bank-conflict, or coalescing measurements back the per-SM
  residency claims.
- **Occupancy, bank conflicts, and coalescing:** `gemmOptimized` uses
  `+1`-padded shared arrays (`As[TILE_SIZE][TILE_SIZE + 1]`) to avoid bank
  conflicts and uses explicit `fmaf`. It exists in
  `source/gemm/cuda/gemm.cpp` but is **not dispatched** by `gemmGPU`, so its
  bank-conflict avoidance and row-major coalescing are labeled unimplemented
  alternatives for this analysis. The dispatched `gemmTiled` uses unpadded 16x16
  shared tiles, which can bank-conflict on `Bs[k][tx]` reads, and performs
  row-major loads along `tx`.
- **Launch caveats:** per-call costs (`cudaMalloc`, H2D and D2H `cudaMemcpy`,
  and the kernel launch) sit outside the kernel but inside the harness window.
  For the small pilot, they dominate any GPU kernel work.
- **Fixed-P scaling:** with `P` fixed, `W/P = Theta(MNK)` keeps the MNK growth
  rate. Asymptotic scaling is unchanged by any constant-factor tiling change;
  only `P` (more SMs or resident threads) or a sub-MNK algorithm would move it.
- **Small-pilot degeneracy:** the `W/P` model treats the 12-MAC problem as a
  degenerate case. Its grid is one block, so the roofline is effectively serial,
  not one block per 9 SMs.

## Improvement Options

| Option | Category and promise | Status here |
|---|---|---|
| 2x2 thread tiling / register blocking (`gemmTiled2x2`, TILE_M=32, TILE_N=32, TILE_K=16, THREAD_TILE=2) | Constant-factor improvement in arithmetic intensity and traffic; `W` stays `Theta(MNK)` | Present in `source/gemm/cuda/gemm.cpp` but **not dispatched** from `gemmGPU`; unimplemented alternative, not claimed to help measured numbers |
| Parallel-K split with cross-thread/tree reduction | Parallel-depth improvement (`S` from `Theta(K)` toward `Theta(log K)` or `Theta(K/P)`); costs extra space and an extra reduction sync; `W` stays `Theta(MNK)` | **Not implemented**; hypothesized proposal only |
| Tensor Cores / cuBLAS | Constant-factor speedup with precision, range, and layout constraints (for example FP16/TF32 inputs and alignment) | **Not implemented**, not claimed |
| Sparsity / Strassen | Sub-MNK total-work reduction | **Not claimed.** Valid only if the input contract supplies structure (zeros, block sparsity, or approximate block decomposition), which this harness does not |
| `+1` shared-memory padding and FMA (`gemmOptimized`) | Constant-factor memory-pipeline and instruction-mix improvement | Present but **not dispatched**; unimplemented alternative |

## Correctness Constraints

- **FP32 accumulation order:** the CPU reference performs a strict `i, j, k`
  serial accumulation. The dispatched CUDA and OpenCL kernels accumulate in
  tile-major order (`t` outer, `k` inner), which is a different FP32 rounding
  order. Differences are bounded by tolerance and are validated, not eliminated.
- **Tolerance:** `compareResults` in `source/gemm/main.cpp` uses an absolute
  tolerance of `0.1f`. That is loose relative to typical FP32 GEMM error for
  these magnitudes and inputs, so correctness claims should state the tolerance
  used rather than claiming "exact".
- **`beta * C` semantics:** `C` is both input and output, so repeated calls
  accumulate into their own result. `main.cpp` uses `benchmarkWithReset(...)`
  with a `C_seed` copy, so every timed iteration starts from the same C. This is
  noted explicitly in a comment in `source/gemm/main.cpp`.
- **5-seed validation:** random data is generated for seeds 42, 1337, 2024, 7,
  and 99 over the large pilot. A single random draw was judged insufficient to
  expose boundary behavior at non-multiple-of-16 M, N, and K tile edges. The
  pilots here are all multiples of 16, but the guards `row < M`, `aCol < K`, and
  similar exist for the general case.

## Evidence and Limitations

- This study is source-derived only. No speedup, bandwidth, or occupancy was
  measured or claimed, and numerical performance statements that would require
  benchmark runs are avoided or explicitly labeled as unmeasured.
- The OpenCL C API and C++ wrapper backends were read to confirm that they
  dispatch a `gemmTiled` kernel with the same 16x16 tile and a 2D block shape
  matching `TILE_SIZE`. Their runtime behavior — build success, correctness on
  this machine, achieved throughput — was not exercised during this attempt.
- `gemmTiled2x2` and `gemmOptimized` are labeled unimplemented alternatives
  because the dispatched `gemmGPU` path selects `gemmTiled` exclusively in the
  CUDA backend.
- Any Phase 1 conclusion about GEMM's roofline on real hardware must come from a
  later profiling attempt, not from this study.
