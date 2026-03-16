import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    A = tl.load(a + idx, mask=mask)
    B = tl.load(b + idx, mask=mask)
    tl.store(c + idx, A + B, mask=mask)




def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.dtype == b.dtype, "Input tensors must have the same dtype"

    n_elements = a.numel()
    c = torch.empty_like(a)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    vector_add_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return c



def main():
    device = "cuda"
    n_elements = 1_000_000

    a = torch.rand(n_elements, device=device, dtype=torch.float32)
    b = torch.rand(n_elements, device=device, dtype=torch.float32)

    c_triton = vector_add(a, b)
    c_torch = a + b

    # Verify correctness
    torch.testing.assert_close(c_triton, c_torch)
    print("Triton kernel result matches PyTorch.")

    # Show a small sample
    print("a[:5]      =", a[:5].cpu())
    print("b[:5]      =", b[:5].cpu())
    print("c_triton[:5] =", c_triton[:5].cpu())


if __name__ == "__main__":
    main()
