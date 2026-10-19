# Sigmoid Activation: Complexity Study

> Audience: humans, agents · Applies to: CUDA, OpenCL · Last reviewed: 2026-10

Phase 1, attempt `P1-T3-sigmoid`, based on revision `77038c4`. This study
derives asymptotic work, depth, and traffic bounds for the sigmoid kernel from
source only. No measurements are claimed.

## 1. Operation and Dimensions

Elementwise sigmoid over `n` FP32 elements:

```
sigma(x) = 1 / (1 + exp(-x))
```

- **Harness:** `source/sigmoid/main.cpp` fixes `N = 1024` as a single vector
  (`n = N = 1024`); there is no matrix or multidimensional shape.
- **Data:** one contiguous FP32 input array and one contiguous FP32 output array
  of the same length. The operation is a one-to-one map.
- **Backends:** the CPU path `sigmoid_cpu` and two GPU kernel variants are
  covered below. The OpenCL C API and OpenCL C++ wrapper backends use the same
  two-kernel structure (scalar plus `float4` vectorized), with `native_exp` in
  place of `__expf`.

## 2. Source Links

- Harness and benchmark driver: `source/sigmoid/main.cpp`
- CUDA implementation: `source/sigmoid/cuda/sigmoid.cpp`
- CUDA header (`BSIZE = 256`, kernel prototypes):
  `source/sigmoid/cuda/sigmoid.h`
- OpenCL C API implementation: `source/sigmoid/opencl/sigmoid.cpp`
- OpenCL C++ wrapper implementation: `source/sigmoid/opencl_cpp/sigmoid.cpp`
- Shared benchmark utility: `source/utils/benchmark.h`

## 3. CPU Work

`sigmoid_cpu` (identical in each backend `.cpp`) is a serial loop:

```
for i in 0..N-1: output[i] = 1.0f / (1.0f + expf(-input[i]));
```

- **Total work:** `Theta(n)` — one exponential, one divide, one negate, and one
  add per element, a constant amount of work per iteration.
- **Parallel depth of this serial code:** `Theta(n)` if taken as written, since
  it is a sequential loop. The recurrence itself is elementwise, so it *could*
  be evaluated with `Theta(1)` depth under parallel evaluation, matching the GPU.
- **Traffic:** `Theta(n)` reads plus `Theta(n)` writes. At 4 bytes per FP32
  element, that is 4 bytes in and 4 bytes out per element — 8 bytes total, near
  the lower bound for this operation.

## 4. Actual GPU Work and Depth

### 4.1 Scalar Kernel (`sigmoid_kernel`)

- One thread computes one output:
  `i = blockIdx * blockDim + threadIdx`, guarded by `if (i >= N) return;`.
- **Total work:** `W = Theta(n)`. Each of the `n` threads performs `Theta(1)`
  math: negate, `exp` / `__expf` / `native_exp`, add, divide.
- **Parallel depth:** `S = Theta(1)`. The per-element chain is negate → exp →
  add → divide, a constant number of dependent FP operations. Threads are
  independent: there are no cross-thread dependencies, reductions, or
  synchronization, so depth does not grow with `n`.
- Addressing is block-linear, so global memory accesses are fully coalesced.

### 4.2 Vectorized Kernel (`sigmoid_kernel2`)

- One thread computes four outputs: it loads one `float4`, applies sigmoid to
  `.x`, `.y`, `.z`, and `.w`, and stores one `float4`. Threads with
  `i + 3 >= N` fall back to a scalar loop over the remaining tail elements, at
  most three iterations of constant work — `O(1)` with respect to `n`.
- **Total work:** `W = Theta(n)`. There are four constant-work element
  evaluations per thread across roughly `n/4` threads: the same total work in a
  different grouping.
- **Parallel depth:** `S = Theta(1)`, with the same per-element dependent chain
  and still no cross-thread coupling.
- Both variants are already work-optimal (each element is read once and written
  once with `Theta(1)` math) and depth-optimal. `Theta(n)` work with `Theta(1)`
  depth is the frontier for a streaming elementwise operation.

## 5. Auxiliary Storage and Traffic

Keywords: auxiliary space; memory traffic; finite-resource model.

- **Auxiliary storage:** `O(1)`. There is no shared memory, no scratch buffer,
  and no intermediate array — only kernel arguments and a handful of registers
  (the OpenCL variants also use no `__local` allocations).
- **Minimum traffic:** at least one read and one write per element, or `8n`
  bytes for FP32 (`4n` in and `4n` out). The common host path in `sigmoid_gpu`
  and `sigmoid2_gpu` adds a one-time `cudaMalloc`/`cudaMemcpy` H2D upload and
  D2H download per call, plus a kernel launch. The measured host-path cost
  therefore includes about `4n` bytes host-to-device and `4n` bytes
  device-to-host on top of device-side access. That overhead is constant per
  call, not per element, and both variants have the same asymptotic traffic.
- **Vectorized variant:** contiguous `float4` loads and stores issue as fully
  coalesced 128-byte transactions. The traffic count is unchanged at `8n` bytes
  device-side; the improvement is in transaction efficiency and instruction
  count, a constant-factor effect.

## 6. Finite-Device Model

Keywords: auxiliary space; memory traffic; finite-resource model.

Let `P` be the number of processors in a **resource model**: a budget of compute
and memory resources (SM count × active warps, execution-pipeline throughput,
and DRAM bandwidth), **not** a raw thread count. Thousands of threads may
compete for a much smaller effective `P`.

- **Lower bound:** `T_P >= max(W/P, S) = max(Theta(n/P), Theta(1))`. At this
  problem size, the scalar kernel becomes memory- or launch-bound once `n/P`
  falls below the streaming-bandwidth floor.
- **Brent-style upper bound:** `T_P = O(W/P + S) = O(n/P + 1)`. It holds under
  assumptions that are satisfied here but worth restating:
  - balanced, uniform per-element data — the inputs are a contiguous array and
    the guard keeps threads aligned;
  - enough independent work to fill `P` — `n` must exceed the launch and
    occupancy ramps, and at `N = 1024` the kernel may underfill a large device;
  - no serialization through memory — satisfied by independent addresses and no
    atomics or contention.
- **Caveats:**
  - **Bandwidth bound:** the expected bottleneck for a pure elementwise stream.
    Arithmetic intensity is approximately 1 FLOP per 8 bytes moved.
  - **Launch fixed cost:** one kernel launch plus, on the harness path, two
    `cudaMalloc`, two `cudaMemcpy`, and two `cudaFree` calls per timed call
    dominate at `n = 1024`.
  - **Occupancy:** `BSIZE = 256` threads per block gives few blocks at this `N`,
    and a small grid limits the GPU's ability to amortize latency through many
    concurrent warps.

## 7. Improvement Options

Asymptotic work (`Theta(n)`) and depth (`Theta(1)`) are **already optimal**; no
option can reduce the dominant terms below `n/1`. The remaining gains are
constant-factor or traffic improvements:

1. **Constant factor — vectorization.** The scalar kernel processes one element
   per thread; `sigmoid_kernel2` (float4 with `__restrict__`) already groups
   four. Vectorizing the scalar path changes transaction efficiency and
   instruction throughput per element: a fixed-factor gain, and an
   unimplemented hypothesis for that path.
2. **Constant factor — fast-math exponential.** Replacing `expf` with `__expf`
   (CUDA) or `native_exp` (OpenCL) lowers per-element ALU latency and
   instruction count at an accuracy cost. `sigmoid_kernel2` already uses
   `__expf`; the scalar CUDA kernel still uses accurate `exp`. Applying the same
   mapping to the scalar kernel is a constant-factor change only.
3. **Memory traffic — fusion.** Applying sigmoid directly inside a producing
   kernel, such as a GEMM epilogue (for example a cuBLASLt `gemmLt`
   beta/`gemmGemmEx` custom epilogue) or a fused pre-activation, would remove
   the extra read and write round-trip. Traffic would drop from at least `8n`
   bytes streamed to a partial read and write in an already-hot output. This is
   a **traffic improvement and an unimplemented hypothesis**: no fused kernel
   exists in `source/`.
4. **Memory space — buffer reuse.** The host wrappers allocate and free device
   buffers and copy H2D/D2H on every call. A persistent device buffer with
   pinned host staging would cut per-call fixed overhead. This shifts work
   between memory spaces and reduces allocation churn; it is not an asymptotic
   gain.
5. **Parallel depth — not applicable.** `S` is already `Theta(1)`, so there is
   nothing to reduce.

## 8. Correctness Constraints

- **Numerical range:** `exp(-x)` becomes `INF` for sufficiently negative `x` on
  some builds. `1/(1 + inf) = 0` is the mathematically desired saturation, but
  the written form relies on that behavior rather than clamping explicitly.
  Very large positive `x` forces `exp(-x)` toward 0 and yields exactly 1.0. Both
  are correct saturated sigmoid values, so the behavior is fine but worth
  noting.
- **Harness tolerance:** the absolute difference against the CPU reference must
  be at most `1e-5` per element, checked in `source/sigmoid/main.cpp` for both
  kernels.
- **Accuracy of the vectorized kernel:** `__expf` (CUDA) and `native_exp`
  (OpenCL) are fast intrinsics with larger ULP error. They are not IEEE-close,
  and CUDA documents `__expf(x)` as roughly comparable to `expf` for many
  ranges but not guaranteed to match the accurate path. The `1e-5` tolerance
  has been shown by the harness to hold for the `N = 1024` test spread
  `(i % 100) - 50.0f` in `[-50, 49]`, where sigmoid saturates outside a narrow
  linear window — far from the denormal and underflow region where intrinsic
  error could grow.
- **Tolerance sensitivity:** if inputs range past the saturation window, or the
  tolerance is tightened, the fast-math intrinsics may violate the check. The
  vectorized tail branch already uses `__expf`, so tightening the tolerance
  would require re-validating both backends.

## 9. Evidence and Limitations

- **Evidence basis:** every claim above is derived from source under
  `source/sigmoid/` at revision `77038c4` — the kernel bodies, launch
  configuration (`BSIZE = 256`), host-wrapper allocation pattern, and the
  harness's `N = 1024` and `1e-5` tolerance. There was no profiling, no
  measurement, and no hardware assumption beyond the abstract `P`-processor
  model.
- **Fixed harness size:** `main.cpp` hardcodes `N = 1024`. No sweep across `n`
  was run, and this packet is documentation-only, so the asymptotic statements
  are *bounds derived from the code*, not empirical performance claims. At
  `n = 1024` the kernel almost certainly sits in the launch- and
  setup-dominated regime, so measured behavior would not resemble
  asymptotic-regime scaling.
- **Not covered here:** measured speedups, real device bandwidth ceilings,
  `cuBLASLt`-style epilogue feasibility, and any runtime comparison between
  OpenCL and CUDA.
- No `docs/algorithmic-complexity.md` exists at this revision. This document
  follows the section list in the Phase 1 packet and is intended to seed that
  collection.
