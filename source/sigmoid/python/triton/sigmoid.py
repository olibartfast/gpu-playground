"""Triton backend for the sigmoid kernel.

What: elementwise sigmoid y = 1 / (1 + exp(-x)) over a 1-D float32 tensor,
computed by a single Triton kernel.

Run: `python source/sigmoid/python/triton/sigmoid.py [--performance]` from
the repo root.

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
def sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)


def solve(X: torch.Tensor, Y: torch.Tensor, N: int):
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    sigmoid_kernel[grid](X, Y, N, BLOCK_SIZE)


def _validate() -> bool:
    torch.manual_seed(0)
    ok = True
    for n in (1, 1000, LARGE_N):
        X = torch.randn(n, device="cuda", dtype=torch.float32)
        Y = torch.empty_like(X)
        solve(X, Y, n)
        expected = torch.sigmoid(X)
        ok &= check_close(Y, expected, ATOL, RTOL, name=f"n={n}")
    return ok


def main() -> int:
    ok = _validate()

    bench_n = PERF_N if "--performance" in sys.argv else LARGE_N
    X = torch.randn(bench_n, device="cuda", dtype=torch.float32)
    Y = torch.empty_like(X)

    r = benchmark(lambda: solve(X, Y, bench_n))
    print_bench("GPU (triton):", r)

    rt = benchmark(lambda: torch.sigmoid(X))
    print_bench("PyTorch:", rt)
    print_speedup("Speedup (PyTorch/Triton):", rt, r)

    bytes_moved = 2 * bench_n * X.element_size()
    gbps = (bytes_moved / 1e9) / (r.median_ms / 1000.0)
    print(f"Bandwidth: {gbps:.2f} GB/s (n={bench_n})")

    print("Overall result: PASSED" if ok else "Overall result: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
