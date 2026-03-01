import torch
import triton
import triton.language as tl


# ----------------------------
# Triton Kernel
# ----------------------------
@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)


def solve(X: torch.Tensor, Y: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    sigmoid_kernel[grid](X, Y, N, BLOCK_SIZE)


# ----------------------------
# Validation + Benchmark
# ----------------------------
def validate_and_benchmark():
    assert torch.cuda.is_available(), "CUDA not available"

    device = "cuda"
    dtype = torch.float32
    N = 10_000_000

    X = torch.randn(N, device=device, dtype=dtype)
    Y = torch.empty_like(X)

    # ----------------------------
    # ✅ Validation
    # ----------------------------
    solve(X, Y, N)
    torch.cuda.synchronize()

    torch_result = torch.sigmoid(X)
    max_error = torch.max(torch.abs(Y - torch_result)).item()

    print(f"Max error: {max_error:.6e}")
    if max_error < 1e-6:
        print("Validation: ✅ PASSED")
    else:
        print("Validation: ❌ FAILED")

    # ----------------------------
    # ✅ Warmup
    # ----------------------------
    for _ in range(10):
        solve(X, Y, N)
        torch.sigmoid(X)
    torch.cuda.synchronize()

    # ----------------------------
    # ✅ Benchmark (averaged)
    # ----------------------------
    runs = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Triton timing
    start.record()
    for _ in range(runs):
        solve(X, Y, N)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / runs  # ms

    # PyTorch timing
    start.record()
    for _ in range(runs):
        torch.sigmoid(X)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / runs  # ms

    # ----------------------------
    # ✅ Bandwidth Calculation
    # ----------------------------
    bytes_processed = 2 * X.numel() * X.element_size()  # read + write
    gb = bytes_processed / 1e9

    triton_bw = gb / (triton_time / 1000)
    torch_bw = gb / (torch_time / 1000)

    # ----------------------------
    # ✅ Results
    # ----------------------------
    print("\n--- Benchmark Results ---")
    print(f"Triton time:  {triton_time:.3f} ms")
    print(f"PyTorch time: {torch_time:.3f} ms")
    print(f"Speedup:      {torch_time / triton_time:.2f}x")
    print(f"Triton BW:    {triton_bw:.2f} GB/s")
    print(f"PyTorch BW:   {torch_bw:.2f} GB/s")


if __name__ == "__main__":
    validate_and_benchmark()
