# Running on AMD Developer Cloud

> Audience: humans, agents · Applies to: all · Last reviewed: 2026-10

Use a ROCm-enabled AMD GPU Droplet (AMD Developer Cloud, built on DigitalOcean
infrastructure) to build the HIP implementations in
`source/<kernel>/hip/`. The shared C++ harnesses compare CPU and GPU results.
CUDA remains the default backend; HIP is selected with `USE_HIP=ON`.

## Connect and inspect the instance

Provision an AMD GPU Droplet (`gpu-mi300x1-192gb`) with ROCm development tools already installed.
The provider console supplies the IP address and login user. From your laptop:

```bash
export AMD_HOST=root@YOUR_INSTANCE_IP
export AMD_KEY="$HOME/.ssh/id_ed25519"
ssh -o IdentitiesOnly=yes -i "$AMD_KEY" "$AMD_HOST"
```

`IdentitiesOnly=yes` avoids `Too many authentication failures` when the SSH
agent holds multiple keys. On the instance, inspect the tools before building:

```bash
hostname
rocm-smi
command -v hipcc
hipconfig --version
/opt/rocm/bin/rocm_agent_enumerator
cmake --version
/opt/rocm/llvm/bin/clang++ --version
```

HIP requires CMake 3.21+, a C++17 compiler, and the ROCm development toolchain.
Use ROCm's `clang++` as `CMAKE_HIP_COMPILER`, rather than the `hipcc` wrapper.
If ROCm is installed elsewhere, substitute its actual prefix below. No CUDA
toolkit, NVIDIA driver, Ninja, or Python DSL dependencies are required.

## Copy the working sources

From the local repository root, package the current files, including uncommitted
HIP sources. Choose a fresh remote directory to preserve existing checkouts:

```bash
tar -czf /tmp/gpu-playground-hip.tar.gz \
  CMakeLists.txt CMakePresets.json cmake source
scp -o IdentitiesOnly=yes -i "$AMD_KEY" \
  /tmp/gpu-playground-hip.tar.gz "$AMD_HOST":/tmp/
ssh -o IdentitiesOnly=yes -i "$AMD_KEY" "$AMD_HOST"
```

On the instance:

```bash
mkdir -p /root/gpu-playground-hip-run
tar -xzf /tmp/gpu-playground-hip.tar.gz -C /root/gpu-playground-hip-run
cd /root/gpu-playground-hip-run
```

Use a directory under your home when logging in as a non-root user. The archive
contains the build inputs, not a Git checkout. To record the source version,
save `git rev-parse HEAD` and `git diff` locally alongside the archive; keep the
archive itself when it contains untracked files.

## Configure, build, and validate

For MI300X, the architecture is `gfx942`:

```bash
cmake --preset hip \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build --preset hip -j8 > build/hip/build.log 2>&1
ctest --preset hip > build/hip/ctest.log 2>&1
cat build/hip/ctest.log
```

Run each command only after the preceding command succeeds. For another GPU,
use its architecture from `rocm_agent_enumerator`; omit the architecture option
to let CMake detect the local device. The preset uses Unix Makefiles and
`RelWithDebInfo`. Restricting parallelism to eight jobs bounds compiler memory
use. HIP and the two OpenCL options are mutually exclusive.

There are 20 HIP targets and 19 automatic correctness tests. The CNN exercise
`deep_learning_inference` is built but needs an input image and weights for its
manual correctness mode; see its [README](../../source/deep_learning_inference/README.md).
Python Triton and CuTe DSL tests are not registered in HIP builds.

Run an individual target or the larger benchmark cases:

```bash
./build/hip/source/gemm/gemm
./build/hip/source/fp16_dot_product/fp16_dot_product --performance
./build/hip/source/categorical_cross_entropy/categorical_cross_entropy --performance
./build/hip/source/gaussian_blur/gaussian_blur --performance
```

Timings use the existing warm-up/repeated-sample harness. `GPU kernel:` uses
HIP events; `GPU end-to-end:` also includes allocation, transfers, and host
work. Compare like-for-like timing paths. Small functional cases validate
correctness and do not establish an AMD performance advantage.

The shuffle reductions in SpMV and categorical cross-entropy explicitly use
32-lane logical groups, including on 64-lane AMD wavefronts. HIP sources use
`HIP_CHECK` for runtime calls and launch checks. The CNN candidate retains the
original exercise's reference fallback; matching it does not establish an
independently optimized inference implementation.

## Optional CNN smoke check without training dependencies

In a fresh transferred source tree, generate synthetic weights and a tiny PPM
image using the Python standard library. This creates `weights.bin`; use trained
weights instead when validating model quality.

```bash
python3 - <<'PY'
from pathlib import Path
import struct

count = 32*4*3*3 + 32*32 + 256*32 + 1 + 16*256
Path('source/deep_learning_inference/weights.bin').write_bytes(
    struct.pack('<f', 0.001) * count)
pixels = bytes((i * 17) % 256 for i in range(128 * 128 * 3))
Path('build/hip/cnn-smoke.ppm').write_bytes(b'P6\n128 128\n255\n' + pixels)
PY
./build/hip/source/deep_learning_inference/deep_learning_inference \
  correctness build/hip/cnn-smoke.ppm > build/hip/cnn-smoke.log 2>&1
cat build/hip/cnn-smoke.log
```

This checks that both CNN entry points execute and agree. The candidate delegates
to the reference, so this is a runtime smoke check, not independent numerical
validation against PyTorch or evidence of trained model quality.

## Verified session: 2026-10-23

The run used `root@YOUR_INSTANCE_IP`, an AMD Instinct MI300X VF (`gfx942`), CMake 3.28.3,
HIP `7.16.26385-0000000`, and AMD clang `24.0.0git`. The installed path was
`/opt/rocm/llvm/bin/clang++`; the hostname/image label mentioned ROCm 10.1,
which should not be confused with the version reported by `hipconfig`.
Sources and logs were placed under `/root/gpu-playground-hip-20261023`.
The validated build-input archive has SHA-256
`c39b833122e1d144dab4811adbedff0eb4b5303eeba92a5ec06eefb56043b78d`.

- All 20 HIP targets built in `RelWithDebInfo`.
- All 19 CTest correctness checks passed on the GPU.
- The synthetic CNN smoke check passed; trained-weight/PyTorch parity was unrun.
- All three `--performance` cases passed their numerical comparisons.

Observed medians from that session, not guarantees for other instances:

| Case | Shape | HIP kernel | End-to-end |
|------|-------|------------|------------|
| FP16 dot product | 100,000,000 elements | 0.3275 ms | 1425.1499 ms |
| Cross-entropy | N=10,000, C=1,000 | 0.1203 ms | 1.0378 ms |
| Gaussian blur | 512×512, 7×7 | 0.0244 ms | 0.2203 ms |

Device medians use 3 warm-up and 10 timed iterations. FP16 end-to-end includes
host-side FP16 conversion, explaining its large gap from device time. These
measurements establish a HIP baseline; no CUDA-versus-HIP performance claim is
made.

## Collect results and release the instance

From the laptop:

```bash
mkdir -p /tmp/amd-developer-cloud-results
scp -o IdentitiesOnly=yes -i "$AMD_KEY" \
  "$AMD_HOST":/root/gpu-playground-hip-run/build/hip/build.log \
  "$AMD_HOST":/root/gpu-playground-hip-run/build/hip/ctest.log \
  /tmp/amd-developer-cloud-results/
```

Keep the GPU model, architecture, compiler version, source archive, build type,
and command with any benchmark results. Stop or destroy the instance in the
provider console after collecting results; ending SSH alone leaves it running.

## Troubleshooting

- **HIP compiler cannot be identified:** verify the ROCm LLVM path and development
  installation; start with a fresh build directory after changing compilers.
- **No GPU agents:** check `rocm-smi`, driver readiness, and access to `/dev/kfd`
  and `/dev/dri`. A successful build does not prove device execution works.
- **Wrong device architecture:** use the detected `gfx...` value; do not reuse
  CUDA `sm_...` architecture values.
- **A test fails:** retain `build/hip/Testing/Temporary/LastTest.log` and rerun
  that binary directly. Do not treat a skipped or unrun check as a pass.
