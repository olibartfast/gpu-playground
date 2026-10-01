"""CuTe DSL backend for the vector_addition kernel.

What: elementwise vector addition C = A + B over 1-D float32 tensors,
computed by a single CuTe DSL kernel.

Run: `python source/vector_addition/python/cute-dsl/vector_addition.py
[--performance]` from the repo root. Requires the CUTLASS DSL package
(`nvidia-cutlass-dsl`); if it is missing this script prints SKIPPED and
exits 77.

Timing boundary: timings are device-resident: inputs are already on the GPU
and each timed call is launch + kernel only. The `solve` function is compiled
once with `cute.compile` before timing starts, so the timed calls invoke the
compiled callable directly and skip JIT tracing / compile-cache lookup. This
is comparable to the C++ `GPU kernel:` line (benchmarkDevice().device), not
the C++ end-to-end lines that include host/device transfers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils" / "python"))
from gpu_bench import benchmark, check_close, print_bench, print_speedup, require  # noqa: E402

require("torch", "cutlass")

import torch  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

ATOL = 1e-6
RTOL = 1e-6
LARGE_N = 2_000_000
PERF_N = 20_000_000


@cute.kernel
def vec_add_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()

    idx = tx + bx * bdx

    if idx < N:
        C[idx] = A[idx] + B[idx]


@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    threads = 256
    blocks = (N + threads - 1) // threads

    vec_add_kernel(A, B, C, N).launch(
        grid=(blocks, 1, 1),
        block=(threads, 1, 1),
    )


def _run(n: int):
    """Allocate torch tensors, wrap them for CuTe, run solve, return torch tensors."""
    a_torch = torch.rand(n, dtype=torch.float32, device="cuda")
    b_torch = torch.rand(n, dtype=torch.float32, device="cuda")
    c_torch = torch.zeros(n, dtype=torch.float32, device="cuda")

    a = from_dlpack(a_torch)
    b = from_dlpack(b_torch)
    c = from_dlpack(c_torch)

    solve(a, b, c, n)
    return a_torch, b_torch, c_torch


def _validate() -> bool:
    torch.manual_seed(0)
    ok = True
    for n in (1, 1000, LARGE_N):
        a_torch, b_torch, c_torch = _run(n)
        expected = a_torch + b_torch
        ok &= check_close(c_torch, expected, ATOL, RTOL, name=f"n={n}")
    return ok


def main() -> int:
    cutlass.cuda.initialize_cuda_context()

    ok = _validate()

    bench_n = PERF_N if "--performance" in sys.argv else LARGE_N
    a_torch = torch.rand(bench_n, dtype=torch.float32, device="cuda")
    b_torch = torch.rand(bench_n, dtype=torch.float32, device="cuda")
    c_torch = torch.zeros(bench_n, dtype=torch.float32, device="cuda")
    a = from_dlpack(a_torch)
    b = from_dlpack(b_torch)
    c = from_dlpack(c_torch)

    compiled = cute.compile(solve, a, b, c, bench_n)
    r = benchmark(lambda: compiled(a, b, c, bench_n))
    print_bench("GPU (cute-dsl):", r)

    rt = benchmark(lambda: a_torch + b_torch)
    print_bench("PyTorch:", rt)
    print_speedup("Speedup (PyTorch/CuTe):", rt, r)

    bytes_moved = 3 * bench_n * a_torch.element_size()
    gbps = (bytes_moved / 1e9) / (r.median_ms / 1000.0)
    print(f"Bandwidth: {gbps:.2f} GB/s (n={bench_n})")

    print("Overall result: PASSED" if ok else "Overall result: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
