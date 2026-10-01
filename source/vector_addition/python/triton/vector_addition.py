"""Triton backend for the vector_addition kernel.

What: elementwise vector addition c = a + b over 1-D float32 tensors,
computed by a single Triton kernel.

Run: `python source/vector_addition/python/triton/vector_addition.py
[--performance]` from the repo root.

Timing boundary: timings are device-resident: inputs are already on the GPU
and each timed call is launch + kernel only. This is comparable to the C++
`GPU kernel:` line (benchmarkDevice().device), not the C++ end-to-end lines
that include host/device transfers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils" / "python"))
from gpu_bench import benchmark, check_close, print_bench, print_speedup, require  # noqa: E402

require("torch", "triton")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ATOL = 1e-6
RTOL = 1e-6
BLOCK_SIZE = 1024
LARGE_N = 2_000_000
PERF_N = 20_000_000


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    A = tl.load(a + idx, mask=mask)
    B = tl.load(b + idx, mask=mask)
    tl.store(c + idx, A + B, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, n_elements: int):
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.dtype == b.dtype, "Input tensors must have the same dtype"

    c = torch.empty_like(a)
    solve(a, b, c, a.numel())
    return c


def _validate() -> bool:
    torch.manual_seed(0)
    ok = True
    for n in (1, 1000, LARGE_N):
        a = torch.rand(n, device="cuda", dtype=torch.float32)
        b = torch.rand(n, device="cuda", dtype=torch.float32)
        c = vector_add(a, b)
        expected = a + b
        ok &= check_close(c, expected, ATOL, RTOL, name=f"n={n}")
    return ok


def main() -> int:
    ok = _validate()

    bench_n = PERF_N if "--performance" in sys.argv else LARGE_N
    a = torch.rand(bench_n, device="cuda", dtype=torch.float32)
    b = torch.rand(bench_n, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    r = benchmark(lambda: solve(a, b, c, bench_n))
    print_bench("GPU (triton):", r)

    rt = benchmark(lambda: a + b)
    print_bench("PyTorch:", rt)
    print_speedup("Speedup (PyTorch/Triton):", rt, r)

    bytes_moved = 3 * bench_n * a.element_size()
    gbps = (bytes_moved / 1e9) / (r.median_ms / 1000.0)
    print(f"Bandwidth: {gbps:.2f} GB/s (n={bench_n})")

    print("Overall result: PASSED" if ok else "Overall result: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
