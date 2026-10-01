# Validation specified before implementation

Frozen acceptance command (run from the repo root; exits nonzero on any failure):

```bash
specs/2026-10-01-python-backend/acceptance/check.sh
```

| Check | Requirements | Observable evidence |
|---|---|---|
| V-1 | R-2 | `acceptance/helper_check.py` passes. It checks `summarize` for odd, even, single and empty sample lists against hand-computed values; that `benchmark` runs `fn` warmup+iterations times and reports 3+10 by default; that `iterations=0` is clamped to 1; that `check_close` returns True/False without raising on matching/mismatching CUDA tensors; that `print_bench` output contains `(3 warm-up + 10 timed)`; that `print_speedup` prints `n/a` for a zero median; that `require("no_such_module_xyz")` exits with code 77; and that `gpu_bench.py` does not import triton or cutlass |
| V-2 | R-1, R-3, R-5 | `python source/sigmoid/python/triton/sigmoid.py` and `python source/vector_addition/python/triton/vector_addition.py` exit 0 and print `Overall result: PASSED`, at least three validation lines (sizes 1, non-multiple, large), and `(3 warm-up + 10 timed)` |
| V-3 | R-3 | Fault injection: acceptance mirrors `source/sigmoid/python/triton/sigmoid.py` into a temp tree, with `source/utils` symlinked so the D-3 path still resolves. It perturbs the kernel's `y = 1.0 / (1.0 + tl.exp(-x))` line by `+ 1e-3` and expects exit 1 and `Overall result: FAILED`. The kernel line must remain textually unchanged (R-5) |
| V-4 | R-3, D-4 | CuTe DSL script: `python -m py_compile` passes. If `cutlass` is missing, running it exits 77 with `SKIPPED: cutlass not installed`; if present, it must meet V-2. The outcome is recorded either way |
| V-5 | R-4, R-6 | Each migrated script's module docstring states the device-resident timing boundary; `requirements.txt` exists and lists torch and triton |
| V-6 | R-7 | AGENTS.md, Readme.md and `docs/adding-a-new-kernel.md` mention `python/<dsl>/`, `gpu_bench.py` and exit code 77; the Readme has a `vector_addition` row; `specs/tech-stack.md` mentions Python DSL backends; a manual read confirms the API names match R-2 |
| V-7 | all | Read-only reviewer verdict on the diff against this packet; changed paths are limited to those owned in implementation-workflow.md |

No C++ build is needed: no C++ file changes. If a reviewer finds any C++ change, that alone fails V-7.

## Evidence

- 2026-10-01, T-0 (acceptance assets frozen): `bash -n check.sh` passes. On the
  pre-implementation tree, `check.sh` exits 1 at the py_compile step
  (`gpu_bench.py` missing), as expected.
  - `helper_check.py` passes (`PASSED (0 failures)`) against a throwaway
    reference `gpu_bench.py` kept outside the repo in the session scratchpad.
    That shows V-1 can pass; it is not evidence for the implementation.
  - Environment: torch 2.12.0+cu130, triton 3.7.0, CUDA available; `cutlass` not installed.
- V-1..V-7 against the implementation: not started.
