"""Benchmarking helper shared by Python DSL backends (Triton, CuTe DSL).

Mirrors the protocol in source/utils/benchmark.h: a few untimed warm-up calls
absorb one-off costs (here, JIT compilation) before repeated timed iterations
are reduced to median/mean/min/max/stddev, compared on the median. Timings
are device-resident: each timed call is bracketed by a fresh pair of
torch.cuda.Event(enable_timing=True) on the current stream, so the result is
comparable to the C++ `GPU kernel:` line (benchmarkDevice().device), not the
end-to-end lines that include host/device transfers.

Depends on torch only; never imports triton or cutlass.
"""
import importlib
import statistics
import sys
from dataclasses import dataclass

import torch

WARMUP = 3
ITERATIONS = 10


@dataclass
class BenchResult:
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    stddev_ms: float = 0.0
    warmup: int = 0
    iterations: int = 0


def summarize(samples: list[float], warmup: int = 0) -> BenchResult:
    """Reduce timing samples (ms, warm-up already excluded) to a BenchResult.

    Order-independent. Median is the middle element for an odd count and the
    mean of the two middle elements for an even count. Stddev is the sample
    stddev (n-1), 0 for a single sample. Empty `samples` yields a zeroed
    result (iterations == 0), with `warmup` still recorded.
    """
    result = BenchResult(warmup=warmup)
    n = len(samples)
    if n == 0:
        return result
    result.iterations = n

    ordered = sorted(samples)
    result.min_ms = ordered[0]
    result.max_ms = ordered[-1]
    mid = n // 2
    result.median_ms = ordered[mid] if n % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])
    result.mean_ms = sum(ordered) / n
    result.stddev_ms = statistics.stdev(ordered) if n > 1 else 0.0
    return result


def benchmark(fn, warmup: int = WARMUP, iterations: int = ITERATIONS) -> BenchResult:
    """Time `fn` with CUDA events: `warmup` untimed calls, then `iterations` timed."""
    if iterations < 1:
        iterations = 1
    if warmup < 0:
        warmup = 0

    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    return summarize(samples, warmup)


def print_bench(label: str, r: BenchResult) -> None:
    """Print `r` in the same format as the C++ printBench()."""
    print(f"{label:<22} median {r.median_ms:9.4f} ms | mean {r.mean_ms:9.4f} ms "
          f"+/- {r.stddev_ms:.4f} | min {r.min_ms:9.4f} | max {r.max_ms:9.4f}  "
          f"({r.warmup} warm-up + {r.iterations} timed)")


def print_speedup(label: str, baseline: BenchResult, candidate: BenchResult) -> None:
    """Print baseline/candidate median speedup, matching the C++ printSpeedup()."""
    if candidate.median_ms > 0.0:
        print(f"{label:<22} {baseline.median_ms / candidate.median_ms:.2f}x "
              f"(median baseline / median candidate)")
    else:
        print(f"{label:<22} n/a (candidate median is 0)")


def check_close(actual, expected, atol: float, rtol: float, name: str = "") -> bool:
    """torch.allclose-semantics check; prints PASS/FAIL and max abs diff, never raises."""
    prefix = f"{name}: " if name else "check_close: "

    if actual.shape != expected.shape:
        print(f"{prefix}FAIL (shape mismatch: actual {tuple(actual.shape)} "
              f"vs expected {tuple(expected.shape)})")
        return False

    if actual.dtype != expected.dtype:
        print(f"{prefix}FAIL (dtype mismatch: actual {actual.dtype} vs expected {expected.dtype})")
        return False

    if actual.device != expected.device:
        print(f"{prefix}FAIL (device mismatch: actual {actual.device} vs expected {expected.device})")
        return False

    try:
        ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        if actual.numel() == 0:
            max_diff = 0.0
        else:
            max_diff = (actual - expected).abs().max().item()
    except Exception as exc:  # noqa: BLE001
        print(f"{prefix}FAIL ({exc})")
        return False

    status = "PASS" if ok else "FAIL"
    print(f"{prefix}{status} (max abs diff {max_diff:.6g})")
    return ok


def require(*modules: str) -> None:
    """Import each module by name; skip (exit 77) with a message on the first miss."""
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            print(f"SKIPPED: {module} not installed")
            sys.exit(77)
