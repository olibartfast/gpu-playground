# GPU Programming Examples

> Audience: humans · Applies to: all · Last reviewed: 2026-10

These examples show the same GPU workload — usually matrix multiplication —
expressed in PyTorch, TinyGrad, Triton, and Mojo. They are standalone
illustrations for comparison and experimentation, not part of the repository's
C++ harnesses. For building the C++/CUDA kernels on Colab, see the
[Google Colab setup guide](../setup/google-colab.md).

## PyTorch

A minimal GPU-accelerated tensor operation:

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Running on CPU")

# Create tensor on GPU
tensor = torch.rand(1000, 1000, device=device)
result = torch.matmul(tensor, tensor)
print("Result on GPU:", result)
```

### Profiling CUDA kernels from Python

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    tensor = torch.rand(1000, 1000, device='cuda')
    result = torch.matmul(tensor, tensor)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## TinyGrad

TinyGrad is a minimalist neural network library with its own GPU runtime.

```python
from tinygrad.tensor import Tensor

# Check GPU availability
print(f"Using GPU: {Tensor.gpu}")

# Create tensors
a = Tensor.rand(1024, 1024)
b = Tensor.rand(1024, 1024)
result = a.matmul(b)
print(result.shape)

# Basic neural network
x = Tensor.rand(128, 10)
W = Tensor.rand(10, 20)
b = Tensor.rand(20)
out = x.matmul(W).add(b)
print(out.shape)
```

### Timing a TinyGrad matrix multiplication

```python
import time
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters

# Enable GPU mode
Tensor.no_gpu = False

# Create large tensors
size = 4096
a = Tensor.rand(size, size)
b = Tensor.rand(size, size)

# Benchmark matrix multiplication
GlobalCounters.reset()
start_time = time.time()
c = a.matmul(b)
c.realize()  # Force computation
end_time = time.time()

print(f"Matrix multiplication {size}x{size} took {end_time - start_time:.4f} seconds")
print(f"FLOPS: {2 * size**3 / (end_time - start_time):.2e}")
print(GlobalCounters.print_hot())
```

## Triton

Triton exposes a tile-based programming model for GPU kernels. This example
implements a blocked matrix multiplication with boundary masks.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Boundary checks for non-aligned sizes
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store with boundary check
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return c
```

### Checking the Triton result

```python
a = torch.randn((1024, 1024), device='cuda', dtype=torch.float32)
b = torch.randn((1024, 1024), device='cuda', dtype=torch.float32)
c_torch = torch.matmul(a, b)
c_triton = matmul(a, b)
print(f"Max difference: {(c_torch - c_triton).abs().max().item()}")
```

### Benchmarking Triton against PyTorch

```python
import time

sizes = [512, 1024, 2048, 4096]
torch_times = []
triton_times = []

for size in sizes:
    a = torch.randn((size, size), device='cuda', dtype=torch.float32)
    b = torch.randn((size, size), device='cuda', dtype=torch.float32)

    # Warmup
    torch.matmul(a, b)
    matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    start = time.time()
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 10
    torch_times.append(torch_time)

    # Benchmark Triton
    start = time.time()
    for _ in range(10):
        matmul(a, b)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 10
    triton_times.append(triton_time)

    print(f"Size: {size}x{size}")
    print(f"PyTorch: {torch_time*1000:.2f} ms")
    print(f"Triton: {triton_time*1000:.2f} ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    print("-" * 30)
```

## Mojo (Conceptual)

> **Note:** This is a conceptual example of Mojo's GPU programming patterns, not
> a supported build target. Mojo's GPU support is evolving, so the syntax may
> differ from the current SDK. It requires a local Mojo SDK installation; unlike
> the C++/CUDA code in this repository, it cannot run on Google Colab.

```mojo
from math import div_ceil
from memory import memset_zero
from sys.intrinsics import strided_load
from sys import num_physical_cores
from algorithm import parallelize
from time import now

struct Matrix:
    var rows: Int
    var cols: Int
    var data: DTypePointer[DType.float32]

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    fn __del__(owned self):
        self.data.free()

    @staticmethod
    fn rand(rows: Int, cols: Int) -> Matrix:
        let m = Matrix(rows, cols)
        for i in range(rows * cols):
            m.data[i] = Float32(randf())
        return m

    fn matmul(self, other: Matrix) -> Matrix:
        let result = Matrix(self.rows, other.cols)
        @parameter
        fn p(i: Int):
            for j in range(result.cols):
                var sum = Float32(0.0)
                for k in range(self.cols):
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j]
                result.data[i * result.cols + j] = sum
        parallelize[p](result.rows, num_physical_cores())
        return result

fn matmul_gpu(a: Matrix, b: Matrix) -> Matrix:
    let M = a.rows
    let N = b.cols
    let K = a.cols
    let result = Matrix(M, N)

    @cuda
    fn matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
        let tid_x = cuda.thread_idx_x()
        let tid_y = cuda.thread_idx_y()
        let block_x = cuda.block_idx_x()
        let block_y = cuda.block_idx_y()
        let BLOCK_SIZE = 16
        var sum = Float32(0.0)
        for k in range(0, K):
            sum += a_ptr[block_y * BLOCK_SIZE + tid_y, k] * b_ptr[k, block_x * BLOCK_SIZE + tid_x]
        c_ptr[block_y * BLOCK_SIZE + tid_y, block_x * BLOCK_SIZE + tid_x] = sum

    let grid_x = div_ceil(N, 16)
    let grid_y = div_ceil(M, 16)
    matmul_kernel[(grid_y, grid_x, 1), (16, 16, 1)](a.data, b.data, result.data, M, N, K)
    return result

fn main():
    let size = 1024
    let a = Matrix.rand(size, size)
    let b = Matrix.rand(size, size)

    let start = now()
    let c = a.matmul(b)
    let cpu_time = (now() - start) / 1e9
    print("CPU Matrix multiplication:", cpu_time, "seconds")

    let start_gpu = now()
    let c_gpu = matmul_gpu(a, b)
    let gpu_time = (now() - start_gpu) / 1e9
    print("GPU Matrix multiplication:", gpu_time, "seconds")
    print("Speedup:", cpu_time / gpu_time, "x")
```

## Benchmarking Conventions

The Python snippets above use `time.time()` for brevity. Repository kernels use
the shared timing utilities instead, so measurements include warm-up and report
a distribution rather than a single sample:

```cpp
#include "benchmark.h"

BenchResult gpu = benchmark([&] { run_kernel(); });
double gflops = gpu_benchmark::giga_operations_per_second(operation_count, gpu.median_ms);
double bandwidth = gpu_benchmark::gigabytes_per_second(bytes_moved, gpu.median_ms);

// Kernel-only time (e.g. CUDA events) alongside end-to-end:
DeviceBenchResult device = benchmarkDevice([&] {
    float kernel_ms = 0.0f;
    run_kernel(&kernel_ms);
    return kernel_ms;
});
double kernel_gflops = gpu_benchmark::giga_operations_per_second(operation_count, device.device.median_ms);
```

GPU wrappers must synchronize before returning. Report kernel-only and
end-to-end measurements separately when allocation or transfers are material.

When comparing frameworks, keep the measurement honest:

- Warm up before timing, and time repeated iterations.
- Use batch sizes that fit the available GPU memory.
- Account for host-to-device and device-to-host transfer overhead.
- Handle out-of-memory and other GPU errors explicitly.
- Choose data types deliberately (`float32` versus `float64`).
- Minimize host-device transfers.
- Choose appropriate block and grid sizes, guided by profiling.
