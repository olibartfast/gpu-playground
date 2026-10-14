# Sigmoid Activation — Complexity Analysis (Phase 1, T3)

Attempt ID: `P1-T3-sigmoid`, base revision `77038c4`. This document derives
asymptotic work, depth, and traffic bounds for the sigmoid kernel from source
only. No measurements are claimed.

## 1. Operation and Dimensions

Elementwise sigmoid over `n` FP32 elements:

```
sigma(x) = 1 / (1 + exp(-x))
```

- Harness: `source/sigmoid/main.cpp` fixes `N = 1024` (a single vector,
  `n = N = 1024`); there is no matrix or multi-dimensional shape.
- Input: one contiguous FP32 array; output: one contiguous FP32 array of the
  same length. Operation is purely elementwise (one-to-one map).
- The CPU path `sigmoid_cpu` and two GPU kernel variants are covered below;
  the OpenCL C API and OpenCL C++ wrapper backends use the same two-kernel
  structure (scalar + `float4` vectorized) with `native_exp` in place of
  `__expf`.

## 2. Source Links

- Harness and benchmark driver: `source/sigmoid/main.cpp`
- CUDA implementation: `source/sigmoid/cuda/sigmoid.cpp`
- CUDA header (`BSIZE = 256`, kernel prototypes): `source/sigmoid/cuda/sigmoid.h`
- OpenCL C API implementation: `source/sigmoid/opencl/sigmoid.cpp`
- OpenCL C++ wrapper implementation: `source/sigmoid/opencl_cpp/sigmoid.cpp`
- Shared benchmark utility: `source/utils/benchmark.h`

## 3. CPU Work

`sigmoid_cpu` (in each backend `.cpp`, identical loops) is a serial loop:

```
for i in 0..N-1: output[i] = 1.0f / (1.0f + expf(-input[i]));
```

- Total work: **Theta(n)** — one exponential plus one divide (plus one negate
  and one add) per element, a constant amount of work per iteration.
- Parallel depth of this serial code: Theta(n) if taken as-is (it is a
  sequential loop); the recurrence itself is elementwise, so it *could* be
  evaluated with Theta(1) depth given parallel evaluation, matching the GPU.
- Traffic: Theta(n) reads + Theta(n) writes (8 bytes per element moved in and
  4 out, FP32) — near the lower bound for this operation.

## 4. Actual GPU Work and Depth

### 4.1 Scalar kernel (`sigmoid_kernel`)

- One thread computes one output: `i = blockIdx * blockDim + threadIdx`,
  guarded by `if (i >= N) return;`.
- Total work: **W = Theta(n)** — Theta(1) math per thread (negate, `exp` /
  `__expf` / `native_exp`, add, divide), n threads.
- Parallel depth: **S = Theta(1)** — the per-element chain is negate → exp →
  add → divide (constant FP operations in sequence); threads are
  *independent*, no cross-thread dependencies, no reductions, no
  synchronization. Depth does not grow with n.
- Addressing is block-linear, so global memory accesses are fully coalesced.

### 4.2 Vectorized kernel (`sigmoid_kernel2`)

- One thread computes four outputs: it loads one `float4`, applies sigmoid to
  `.x/.y/.z/.w`, and stores one `float4`. Threads with `i + 3 >= N` fall back
  to a scalar loop over the remaining tail elements (at most 3 iterations of
  constant work — O(1) w.r.t. n).
- Total work: **W = Theta(n)** (4 constant-work element evaluations per
  thread in ~n/4 threads; same total, different grouping).
- Parallel depth: **S = Theta(1)** — same per-element dependent chain, still
  no cross-thread coupling.
- Both variants are already work-optimal (each element is read once, written
  once, with Theta(1) math) and depth-optimal: **Theta(n) work / Theta(1)
  depth** is the frontier for a streaming elementwise op.

## 5. Auxiliary Storage and Traffic

Keywords: auxiliary space; memory traffic; finite-resource model.

- Auxiliary storage: **O(1)** — no shared memory, no scratch buffers, no
  intermediate arrays; only kernel arguments and a handful of scalars in
  registers (the OpenCL variants also carry no `__local` allocations).
- Minimum traffic for this op: at least **1 read + 1 write per element** =
  8n bytes FP32 (4n bytes in + 4n bytes out). The common implementation path
  in `sigmoid_gpu` / `sigmoid2_gpu` adds one-time `cudaMalloc/cudaMemcpy`
  H2D upload and D2H download per call plus a kernel launch, so the measured
  host-path cost includes ~4n bytes host→device and ~4n bytes device→host on
  top of any device-side access; that overhead is constant per call, not
  per element, and both variants have the same asymptotic traffic.
- Vectorized variant: contiguous `float4` loads/stores issue as fully
  coalesced 128-byte transactions — the traffic *count* is the same
  (8n bytes device-side), the improvement is in transaction efficiency and
  instruction count (constant-factor, not asymptotic).

## 6. Finite Device Model

Keywords: auxiliary space; memory traffic; finite-resource model.

Let P be processors in a **resource model** — P is a budget of compute/memory
resources (SM count × active warps, execution-pipeline throughput, and DRAM
bandwidth), **not raw thread count**; thousands of threads may compete for a
much smaller effective P.

- Lower bound: **T_P >= max(W/P, S)** = max(Theta(n/P), Theta(1)) here.
  The scalar kernel is memory-/launch-bound at this problem size once
  n/P falls below the streaming-bandwidth floor.
- Brent-style upper bound: T_P = O(W/P + S) = O(n/P + 1), which holds under
  assumptions that are satisfied here but worth restating:
  - balanced, uniform per-element data (inputs are a straight contiguous
    array; guarding keeps threads aligned),
  - enough independent work to fill P (n must exceed the launch/occupancy
    ramps; at N = 1024 the kernel may underfill a large device),
  - no serialization through memory (satisfied — independent addresses, no
    atomics or contention).
- Caveats:
  - **Bandwidth bound:** expected bottleneck for a pure elementwise stream;
    arithmetic intensity ≈ 1 FLOP per 8 bytes moved.
  - **Launch fixed cost:** one kernel launch plus (in the harness path)
    two `cudaMalloc`, two `cudaMemcpy`, and two `cudaFree` per timed call
    dominates at n = 1024.
  - **Occupancy:** `BSIZE = 256` threads/block, an 8-element-per-block-ish
    grid at this N gives few blocks; small grids limit the GPU's ability to
    amortize latency through many concurrent warps.

## 7. Improvement Options

Asymptotic work (Theta(n)) and depth (Theta(1)) are **already optimal**; no
option can reduce the dominant terms below n/1. Remaining gains:

1. **[constant-factor]** Vectorization: the scalar kernel does 1 element per
   thread; `sigmoid_kernel2` (float4 + `__restrict__`) already groups 4. This
   changes transaction efficiency and instruction throughput per element — a
   fixed-factor gain, unimplemented hypothesis for the scalar path.
2. **[constant-factor]** Fast-math exponential: replacing `expf` with
   `__expf` (CUDA) / `native_exp` (OpenCL) lowers per-element ALU latency and
   instruction count at an accuracy cost. `sigmoid_kernel2` already uses
   `__expf`; the scalar CUDA kernel still uses accurate `exp`. Applying the
   same mapping to the scalar kernel is constant-factor only.
3. **[memory-traffic]** Fusion: sigmoid applied directly inside a producing
   kernel (e.g., GEMM epilogue via cuBLASLt / gemmLt beta/gemmGemmEx
   custom epilogue, or a fused pre-activation) would remove the extra
   read + write round-trip entirely (reduce from ≥ 8n bytes streamed to a
   partial read/write in the already-hot output). This is a **traffic
   improvement, unimplemented hypothesis** in this repo — no fused kernel
   exists in `source/`.
4. **[memory-space]** Buffer reuse across timed iterations: the host wrappers
   allocate/free device buffers and copy H2D/D2H on every call. A persistent
   device buffer + pinned host staging would cut per-call fixed overhead;
   this moves work between memory spaces / host memory (allocation churn),
   again not an asymptotic gain.
5. **[parallel-depth]** Not applicable: S is already Theta(1); there is
   nothing to reduce.

## 8. Correctness Constraints

- Numerical range: `exp(-x) == INF` for sufficiently negative x on some
  builds; `1/(1 + inf) = 0` is mathematically the desired saturation, but the
  written form saturates at 0/1 rather than clamping explicitly. Very large
  positive x forces `exp(-x) → 0`, giving exactly 1.0. Both are the correct
  saturated sigmoid values, so this is behaviorally fine but worth noting.
- Harness tolerance: absolute difference vs. CPU reference must be ≤
  `1e-5` per element, checked in `source/sigmoid/main.cpp` for both kernels.
- Accuracy constraint for the vectorized kernel: `__fexpf`/`__expf`
  (CUDA) and `native_exp` (OpenCL) are fast intrinsics with larger ULP
  error (~medium-low relative error, not IEEE-close; CUDA documents
  `__expf(x)` as comparable to `expf` with roughly 2^-21-ish error for
  many ranges but officially not guaranteed to match the accurate path).
  The `1e-5` tolerance has been shown by the harness to be satisfied for
  the `N = 1024` test spread `(i % 100) - 50.0f` ∈ `[-50, 49]`, where sigmoid
  saturates outside a narrow linear window — far from the denormal/underflow
  region where intrinsic error could grow.
- If inputs ever spread past the saturation range or the tolerance is
  tightened, the fast-math intrinsics may violate the check; falling back to
  accurate `expf` in the vectorized tail-branch already uses `__expf`, so a
  tolerance tightening would require re-validating both backends.

## 9. Evidence and Limitations

- **Evidence basis:** every claim above is derived from source code under
  `source/sigmoid/` at revision `77038c4` — the kernel bodies, launch
  configuration (`BSIZE = 256`), host wrapper allocation pattern, and the
  harness's `N = 1024` / `1e-5` tolerance. No profiling, no measurements, no
  hardware assumptions beyond the abstract P-processor model.
- **Fixed harness size:** `main.cpp` hardcodes `N = 1024`; no sweep across n
  was run and the packet's scope is documentation only, so the asymptotic
  statements are *bounds derived from the code*, not empirical performance
  claims. At n = 1024 the kernel almost certainly sits in the
  launch-/-setup-dominated regime, so measured behavior would not resemble
  the asymptotic-regime scaling anyway.
- **Not covered here:** measured speedups, real device bandwidth ceilings,
  `cuBLASLt`-style epilogue feasibility, and any OpenCL vs. CUDA runtime
  comparison.
- No `docs/algorithmic-complexity.md` exists in the repository at this
  revision; this document was written to the section list given in the Phase 1
  packet and is intended to seed that collection.
