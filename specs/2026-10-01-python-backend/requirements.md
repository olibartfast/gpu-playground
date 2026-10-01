# Requirements: Python DSL backends (Triton, CuTe DSL)

Created: 2026-10-01. Status: implemented and validated 2026-10-01 (CuTe DSL run is a recorded gap; see validation.md).
Branch: `feat/python-backend`, rebased onto `master` at 1f07323 after PR #6 (gaussian_blur)
and PR #7 (device-time benchmarking) merged.

Constraints: [mission](../mission.md), [technical boundaries](../tech-stack.md),
[repository contract](../../AGENTS.md).

## Findings (2026-10-01)

- F-1: three tracked Python kernels already use `source/<kernel>/python/<dsl>/<kernel>.py`:
  - `sigmoid/python/triton/sigmoid.py`
  - `vector_addition/python/triton/vector_addition.py`
  - `vector_addition/python/cute-dsl/vector_addition.py`
- F-2: the layout is undocumented. AGENTS.md lists only `cuda/`, `opencl/` and
  `opencl_cpp/` as backend directories. The Readme kernel table has no Python
  coverage and no `vector_addition` row (that kernel has no C++ code or CMake target).
- F-3: no script returns nonzero on validation failure:
  - `sigmoid.py` prints "FAILED" and exits 0;
  - the CuTe script prints `OK = False` and exits 0;
  - the Triton vector-add script raises on mismatch.
- F-4: timing does not follow the repo protocol:
  - `sigmoid.py` divides one CUDA-event span over 50 back-to-back runs, giving a mean with no spread;
  - both vector-add scripts are not timed at all.
- F-5: each script tests one size. None covers a size that is not a multiple of
  the block, or size 1.
- F-6: entrypoints are inconsistent. `sigmoid.py` and the CuTe script expose a
  LeetGPU-style `solve(...)` on device tensors; the Triton vector add exposes
  `vector_add(a, b) -> c`.
- F-7: environment observed here: torch 2.12.0+cu130 and triton 3.7.0 import,
  with CUDA available. `cutlass` (CuTe DSL) is not installed.
- F-8: the only CI is `.github/workflows/opencl-build.yml`, an Ubuntu runner without a GPU.

## Requirements

- **R-1 — Layout:** a Python backend lives at
  `source/<kernel>/python/<dsl>/<kernel>.py`, with `<dsl>` in {`triton`, `cute-dsl`}.
  The script runs as `python source/<kernel>/python/<dsl>/<kernel>.py [--performance]`
  from the repo root. Python backends are not CMake targets. A kernel may have
  only Python backends (e.g. `vector_addition`).
- **R-2 — Shared helper:** `source/utils/python/gpu_bench.py` depends on torch
  only (no Triton or CUTLASS import). Its API is frozen as:

  ```python
  WARMUP = 3
  ITERATIONS = 10

  @dataclass
  class BenchResult:
      mean_ms: float
      median_ms: float
      min_ms: float
      max_ms: float
      stddev_ms: float
      warmup: int
      iterations: int

  def summarize(samples: list[float], warmup: int = 0) -> BenchResult
  def benchmark(fn, warmup: int = WARMUP, iterations: int = ITERATIONS) -> BenchResult
  def print_bench(label: str, r: BenchResult) -> None
  def print_speedup(label: str, baseline: BenchResult, candidate: BenchResult) -> None
  def check_close(actual, expected, atol: float, rtol: float, name: str = "") -> bool
  def require(*modules: str) -> None
  ```

  `summarize` and the defaults:
  - The statistics match `source/utils/benchmark.h`: the median is the middle
    element for an odd count and the mean of the two middle elements for an even
    count; the sample stddev divides by n-1 and is 0 for one sample.
  - `iterations < 1` is clamped to 1 and `warmup < 0` to 0.

  `benchmark`:
  - runs `fn` untimed `warmup` times; this absorbs JIT compilation;
  - times each iteration with its own pair of `torch.cuda.Event`s on the current
    stream, synchronizing before reading.

  Printing and checks:
  - `print_bench` uses the C++ `printBench` format, including
    `(<w> warm-up + <n> timed)`.
  - `print_speedup` prints `n/a (candidate median is 0)` when the candidate median
    is not positive.
  - `check_close` uses torch.allclose semantics. It prints the label, max abs diff
    and PASS/FAIL, and returns a bool without raising.
  - `require` imports each named module. On the first missing one it prints
    `SKIPPED: <module> not installed` and exits with code 77.
- **R-3 — Script contract:** each Python backend script:
  - calls `require(...)` for its DSL before importing it;
  - exposes `solve(...)` on device tensors, with the LeetGPU signature where the
    challenge defines one;
  - validates against a torch reference with stated `ATOL`/`RTOL` constants, using
    a fixed seed over at least these sizes: 1; a size that is not a multiple of the
    block size; and a large size (≥ 1e6 elements);
  - times `solve` with `benchmark()`, and may also time the torch reference as a
    comparison baseline (`PyTorch:` line plus `print_speedup`);
  - reports bandwidth or throughput computed from `median_ms`;
  - ends with `Overall result: PASSED/FAILED` and `sys.exit(0 if ok else 1)`.

  `--performance` is optional and selects a larger timing shape.
- **R-4 — Timing boundary:** Python timings are device-resident: inputs are
  already on the GPU, and each timed call is launch plus kernel. Each script's
  docstring and the docs state that this is comparable to the C++ `GPU kernel:` line
  (`benchmarkDevice().device`), not to the C++ end-to-end `GPU:` / `GPU end-to-end:`
  lines, which include transfers.
- **R-5 — Migration:** bring all three scripts from F-1 under R-3. Kernel logic
  is unchanged, except that the Triton vector add gains a `solve` entrypoint;
  `vector_add` may remain as a thin wrapper around it.
- **R-6 — Dependencies:** `source/utils/python/requirements.txt` lists `torch`
  and `triton`. The CuTe DSL package is listed as optional in a comment. Versions
  are not pinned; F-7 records the versions validated.
- **R-7 — Docs:**
  - AGENTS.md: backend split, harness contract, Build And Run, and Testing And
    Benchmarking cover R-1–R-4.
  - Readme: a Python backend column or marker in the kernel table, a
    `vector_addition` row, and setup and run commands.
  - `docs/adding-a-new-kernel.md`: an optional Python-backend section and checklist items.
  - `specs/tech-stack.md` lists Python DSL backends.

## Out of scope

- CI for Python. A GPU-less `py_compile` job is a roadmap idea; GPU execution
  needs hardware the CI does not have.
- New Python kernels beyond the three in F-1, and Python ports of other kernels.
- Calling Python from `main.cpp`, or CMake/ctest integration.
- Packaging (`pyproject.toml`, installable modules), or pinned environments.
- Changing the C++ `benchmark.h`. The Python helper mirrors it; kernel-only C++
  timing landed in PR #7 (`specs/2026-10-01-benchmark-device-timing`).

## Decisions and assumptions

- D-1: keep the `python/` level (rather than `source/<kernel>/triton/`). Two
  Python DSLs already exist and share a torch reference, dependencies and the helper.
- D-2: the helper is torch-based, not `triton.testing.do_bench`, so CuTe DSL
  scripts do not depend on Triton. It mirrors the C++ protocol and output format
  so results line up side by side.
- D-3: scripts locate the helper with
  `sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils" / "python"))`,
  without packaging (layout depth fixed by R-1).
- D-4: exit code 77 (the automake "skipped" convention) separates a missing
  dependency from a failure. Acceptance treats 77 as a recorded gap, never as a pass.
- A-1: the CUDA/torch-backed environment is the reference platform. Running CuTe
  DSL scripts requires installing the CUTLASS DSL package; until then they are
  validated for syntax only.
