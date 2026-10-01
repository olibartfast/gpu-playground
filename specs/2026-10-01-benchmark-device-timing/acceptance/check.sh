#!/usr/bin/env bash
# Frozen acceptance for specs/2026-10-01-benchmark-device-timing (V-1..V-7).
# Run from anywhere; exits nonzero on the first failing check.
set -euo pipefail

root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
here="$root/specs/2026-10-01-benchmark-device-timing/acceptance"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cd "$root"

step() { printf '\n== %s\n' "$*"; }
fail() { printf 'ACCEPTANCE FAILED: %s\n' "$*"; exit 1; }

step "V-1/V-4 host-only compile + summarize/benchmarkDevice checks"
g++ -std=c++17 -Wall -Wextra -Werror -I source/utils \
    "$here/summarize_check.cpp" -o "$tmp/summarize_check" || fail "V-1 compile"
"$tmp/summarize_check" || fail "V-1/V-4 assertions"

step "V-2 benchmark.h includes standard headers only"
bad="$(grep -E '^\s*#\s*include' source/utils/benchmark.h | grep -vE '<[a-z_]+>' || true)"
[ -z "$bad" ] || fail "V-2 non-standard include: $bad"

step "V-4 sync contract documented"
grep -qiE 'block|synchroni' source/utils/benchmark.h || fail "V-4 sync-contract comment missing"

step "V-5/V-7 no average_milliseconds outside specs/"
if git grep -n average_milliseconds -- ':!specs'; then fail "V-5 references remain"; fi

step "V-3/V-6 build default preset"
cmake --preset default >/dev/null || fail "configure"
cmake --build --preset default -j"$(nproc)" || fail "build"

bin=build/default/source
run() {  # run <name> [args...] -> saves output to $tmp/<name>.log
    local name="$1"; shift
    "$bin/$name/$name" "$@" >"$tmp/$name.log" 2>&1 || { cat "$tmp/$name.log"; fail "$name $* exited nonzero"; }
}

step "V-3 existing harnesses unchanged"
for k in gemm softmax sigmoid; do
    run "$k"
    grep -q '(3 warm-up + 10 timed)' "$tmp/$k.log" || fail "V-3 $k timing format"
done

step "V-6 migrated harnesses"
for k in fp16_dot_product categorical_cross_entropy; do
    run "$k"
    grep -q 'GPU kernel' "$tmp/$k.log" || fail "V-6 $k missing GPU kernel line"
    grep -q 'GPU end-to-end' "$tmp/$k.log" || fail "V-6 $k missing GPU end-to-end line"
    grep -q '(3 warm-up + 10 timed)' "$tmp/$k.log" || fail "V-6 $k not on benchmark protocol"
done
"$bin/fp16_dot_product/fp16_dot_product" --performance >"$tmp/perf.log" 2>&1 \
    || { cat "$tmp/perf.log"; fail "V-6 fp16_dot_product --performance"; }
run gaussian_blur
grep -q 'Overall result: PASSED' "$tmp/gaussian_blur.log" || fail "V-6 gaussian_blur functional"
"$bin/gaussian_blur/gaussian_blur" --performance >"$tmp/gb_perf.log" 2>&1 \
    || { cat "$tmp/gb_perf.log"; fail "V-6 gaussian_blur --performance"; }
grep -q 'GPU kernel' "$tmp/gb_perf.log" || fail "V-6 gaussian_blur missing GPU kernel line"
grep -q 'GPU end-to-end' "$tmp/gb_perf.log" || fail "V-6 gaussian_blur missing GPU end-to-end line"
[ "$(grep -c '(3 warm-up + 10 timed)' "$tmp/gb_perf.log")" -ge 3 ] \
    || fail "V-6 gaussian_blur kernel time not on benchmark protocol (expect CPU, end-to-end, kernel)"

printf '\nACCEPTANCE PASSED (V-1..V-7 automated; V-7 manual read and V-8 review pending)\n'
