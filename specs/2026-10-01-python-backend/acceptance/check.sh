#!/usr/bin/env bash
# Frozen acceptance for specs/2026-10-01-python-backend (V-1..V-6 automated parts).
# Run from anywhere; exits nonzero on the first failing check.
set -euo pipefail

root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
here="$root/specs/2026-10-01-python-backend/acceptance"
py="${PYTHON:-python3}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cd "$root"

step() { printf '\n== %s\n' "$*"; }
fail() { printf 'ACCEPTANCE FAILED: %s\n' "$*"; exit 1; }

sig=source/sigmoid/python/triton/sigmoid.py
vadd_tr=source/vector_addition/python/triton/vector_addition.py
vadd_cute=source/vector_addition/python/cute-dsl/vector_addition.py

step "syntax: py_compile all Python backends and the helper"
"$py" -m py_compile source/utils/python/gpu_bench.py "$sig" "$vadd_tr" "$vadd_cute" || fail "py_compile"

step "V-1 helper checks"
"$py" "$here/helper_check.py" || fail "V-1"

step "V-2 Triton scripts"
for s in "$sig" "$vadd_tr"; do
    log="$tmp/$(basename "$(dirname "$(dirname "$s")")").log"
    "$py" "$s" >"$log" 2>&1 || { cat "$log"; fail "V-2 $s exited nonzero"; }
    grep -q 'Overall result: PASSED' "$log" || fail "V-2 $s missing Overall result: PASSED"
    grep -q '(3 warm-up + 10 timed)' "$log" || fail "V-2 $s not on benchmark protocol"
    [ "$(grep -ciE 'pass' "$log")" -ge 4 ] || fail "V-2 $s fewer than 3 validation lines"
done

step "V-3 fault injection makes sigmoid fail"
mkdir -p "$tmp/tree/source/sigmoid/python/triton"
ln -s "$root/source/utils" "$tmp/tree/source/utils"
grep -q 'y = 1.0 / (1.0 + tl.exp(-x))' "$sig" || fail "V-3 kernel line changed (R-5)"
sed 's|y = 1.0 / (1.0 + tl.exp(-x))|y = 1.0 / (1.0 + tl.exp(-x)) + 1e-3|' "$sig" \
    >"$tmp/tree/source/sigmoid/python/triton/sigmoid.py"
set +e
"$py" "$tmp/tree/source/sigmoid/python/triton/sigmoid.py" >"$tmp/fault.log" 2>&1
rc=$?
set -e
[ "$rc" -eq 1 ] || { cat "$tmp/fault.log"; fail "V-3 expected exit 1, got $rc"; }
grep -q 'Overall result: FAILED' "$tmp/fault.log" || fail "V-3 missing Overall result: FAILED"

step "V-4 CuTe DSL script"
set +e
"$py" "$vadd_cute" >"$tmp/cute.log" 2>&1
rc=$?
set -e
if [ "$rc" -eq 77 ]; then
    grep -q 'SKIPPED: cutlass not installed' "$tmp/cute.log" || fail "V-4 exit 77 without SKIPPED message"
    echo "GAP: cutlass not installed; CuTe DSL script validated for syntax and skip path only"
elif [ "$rc" -eq 0 ]; then
    grep -q 'Overall result: PASSED' "$tmp/cute.log" || fail "V-4 missing Overall result: PASSED"
    grep -q '(3 warm-up + 10 timed)' "$tmp/cute.log" || fail "V-4 not on benchmark protocol"
else
    cat "$tmp/cute.log"; fail "V-4 CuTe script exited $rc"
fi

step "V-5 timing boundary documented, requirements present"
for s in "$sig" "$vadd_tr" "$vadd_cute"; do
    "$py" - "$s" <<'EOF' || fail "V-5 $s docstring lacks device-resident timing note"
import ast, sys
doc = ast.get_docstring(ast.parse(open(sys.argv[1]).read())) or ""
sys.exit(0 if "device-resident" in doc.lower() else 1)
EOF
done
grep -qE '^torch' source/utils/python/requirements.txt || fail "V-5 torch missing"
grep -qE '^triton' source/utils/python/requirements.txt || fail "V-5 triton missing"

step "V-6 docs mention the contract"
for f in AGENTS.md Readme.md docs/adding-a-new-kernel.md; do
    grep -q 'python/<dsl>' "$f" || fail "V-6 $f missing python/<dsl>/"
    grep -q 'gpu_bench' "$f" || fail "V-6 $f missing gpu_bench"
    grep -q '77' "$f" || fail "V-6 $f missing exit code 77"
done
grep -q 'vector_addition' Readme.md || fail "V-6 Readme missing vector_addition row"
grep -qi 'python' specs/tech-stack.md || fail "V-6 tech-stack missing Python"

printf '\nACCEPTANCE PASSED (automated V-1..V-6; V-6 manual read and V-7 review pending)\n'
