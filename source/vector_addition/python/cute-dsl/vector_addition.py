import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def vec_add_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()

    idx = tx + bx * bdx

    if idx < N:
        C[idx] = A[idx] + B[idx]


@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    threads = 256
    blocks = (N + threads - 1) // threads

    vec_add_kernel(A, B, C, N).launch(
        grid=(blocks, 1, 1),
        block=(threads, 1, 1),
    )


def main():
    cutlass.cuda.initialize_cuda_context()

    N = 1024

    A_torch = torch.arange(N, dtype=torch.float32, device="cuda")
    B_torch = torch.arange(N, dtype=torch.float32, device="cuda") * 2
    C_torch = torch.zeros(N, dtype=torch.float32, device="cuda")

    A = from_dlpack(A_torch)
    B = from_dlpack(B_torch)
    C = from_dlpack(C_torch)

    solve(A, B, C, N)

    expected = A_torch + B_torch

    print("C[:10] =", C_torch[:10].cpu())
    print("OK =", torch.allclose(C_torch, expected))


if __name__ == "__main__":
    main()
