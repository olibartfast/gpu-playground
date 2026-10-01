"""Acceptance check V-1 for specs/2026-10-01-python-backend (frozen asset; workers must not edit).

Exercises source/utils/python/gpu_bench.py against the frozen API in requirements.md R-2.
"""
import contextlib
import io
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HELPER_DIR = ROOT / "source" / "utils" / "python"
sys.path.insert(0, str(HELPER_DIR))

failures = 0


def expect(ok, what):
    global failures
    if not ok:
        print(f"FAIL: {what}")
        failures += 1


def near(a, b):
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)


source = (HELPER_DIR / "gpu_bench.py").read_text()
expect(not re.search(r"^\s*(import|from)\s+(triton|cutlass)\b", source, re.M),
       "gpu_bench.py must not import triton or cutlass")

import torch  # noqa: E402
import gpu_bench as gb  # noqa: E402

expect(gb.WARMUP == 3 and gb.ITERATIONS == 10, "defaults 3 + 10")

odd = gb.summarize([3.0, 1.0, 2.0], 2)
expect(near(odd.median_ms, 2.0) and near(odd.mean_ms, 2.0), "odd median/mean")
expect(near(odd.min_ms, 1.0) and near(odd.max_ms, 3.0), "odd min/max")
expect(near(odd.stddev_ms, 1.0), "odd sample stddev (n-1)")
expect(odd.iterations == 3 and odd.warmup == 2, "odd metadata")
expect(near(gb.summarize([4.0, 1.0, 3.0, 2.0]).median_ms, 2.5), "even median")
one = gb.summarize([5.0])
expect(near(one.median_ms, 5.0) and near(one.stddev_ms, 0.0) and one.iterations == 1, "single sample")
none = gb.summarize([])
expect(none.iterations == 0 and near(none.median_ms, 0.0), "empty samples")

calls = []
x = torch.ones(1024, device="cuda")
r = gb.benchmark(lambda: calls.append((x * 2).sum()))
expect(len(calls) == 13, "fn runs warmup + iterations times")
expect(r.warmup == 3 and r.iterations == 10, "benchmark default metadata")
expect(r.median_ms > 0.0 and r.min_ms <= r.median_ms <= r.max_ms, "benchmark sane timings")
calls.clear()
clamped = gb.benchmark(lambda: calls.append(x + 1), warmup=-1, iterations=0)
expect(clamped.iterations == 1 and clamped.warmup == 0 and len(calls) == 1, "clamping")

a = torch.rand(100, device="cuda")
with contextlib.redirect_stdout(io.StringIO()):
    same = gb.check_close(a, a.clone(), atol=1e-6, rtol=1e-6, name="same")
    diff = gb.check_close(a, a + 1.0, atol=1e-6, rtol=1e-6, name="diff")
expect(same is True, "check_close True on match")
expect(diff is False, "check_close False on mismatch, no raise")

# Added 2026-10-01 after V-7 review (attempt 1): mismatched inputs must FAIL, never raise or broadcast.
def no_raise_false(actual, expected, what):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = gb.check_close(actual, expected, atol=1e-6, rtol=1e-6, name=what)
    except Exception as exc:  # noqa: BLE001
        expect(False, f"check_close raised on {what}: {type(exc).__name__}")
        return
    expect(result is False, f"check_close returns False on {what}")

no_raise_false(torch.zeros(3, device="cuda"), torch.zeros(4, device="cuda"), "shape mismatch")
no_raise_false(torch.zeros(0, device="cuda"), torch.zeros(3, device="cuda"), "empty vs non-empty")
no_raise_false(torch.zeros(2, 1, device="cuda"), torch.zeros(2, device="cuda"), "broadcastable shape mismatch")
no_raise_false(torch.zeros(3, device="cuda"), torch.zeros(3, device="cuda", dtype=torch.float64), "dtype mismatch")
no_raise_false(torch.zeros(3, device="cuda"), torch.zeros(3), "device mismatch")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    gb.print_bench("GPU:", r)
    gb.print_speedup("Speedup:", r, gb.BenchResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0))
out = buf.getvalue()
expect("(3 warm-up + 10 timed)" in out, "print_bench format")
expect("n/a" in out, "print_speedup n/a on zero median")

proc = subprocess.run(
    [sys.executable, "-c",
     f"import sys; sys.path.insert(0, {str(HELPER_DIR)!r}); "
     "import gpu_bench; gpu_bench.require('no_such_module_xyz')"],
    capture_output=True, text=True)
expect(proc.returncode == 77, f"require exits 77 (got {proc.returncode})")
expect("SKIPPED: no_such_module_xyz not installed" in proc.stdout + proc.stderr, "require message")

print(f"{'FAILED' if failures else 'PASSED'} ({failures} failure{'s' if failures != 1 else ''})")
sys.exit(1 if failures else 0)
