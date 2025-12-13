"""
NVFP4 GEMM Kernel - Baseline Implementation
Uses torch._scaled_mm with CPU round-trip (slow but correct)

This is a basic implementation to get you started and on the leaderboard.
Expected performance: ~300-400x slower than speed of light.

To submit:
    popcorn-cli submit --gpu B200 --leaderboard nvfp4_gemm --mode leaderboard nvfp4_gemm_baseline.py
"""

import torch
from task import input_t, output_t
from utils import make_match_reference

# Scaling factor vector size
sf_vec_size = 16


def ceil_div(a, b):
    """Helper function for ceiling division"""
    return (a + b - 1) // b


def to_blocked(scale_vector: torch.Tensor) -> torch.Tensor:
    """
    Convert scale factors to blocked format for torch._scaled_mm
    
    WARNING: This moves data to CPU and back, which is very slow!
    """
    scale_vector_np = scale_vector.cpu().numpy()
    
    # Reshape to blocked format
    m_sf, k_sf = scale_vector_np.shape
    k = k_sf * sf_vec_size
    k_tiles = k // 256
    
    # Create blocked layout
    scale_blocked = scale_vector_np.reshape(m_sf, k_tiles, 256 // sf_vec_size)
    scale_blocked = scale_blocked.transpose(0, 2, 1)
    scale_blocked = scale_blocked.reshape(m_sf * 256 // sf_vec_size, k_tiles)
    
    return torch.from_numpy(scale_blocked).cuda()


def custom_kernel(data: input_t) -> output_t:
    """
    NVFP4 GEMM: C = A @ B.T
    
    Inputs:
        a: M × K × L (NVFP4 - float4_e2m1fn)
        b: N × K × L (NVFP4 - float4_e2m1fn)
        sfa: M × (K/16) × L (FP8 scale factors for A)
        sfb: N × (K/16) × L (FP8 scale factors for B)
        sfa_permuted: Pre-permuted scale factors (not used in baseline)
        sfb_permuted: Pre-permuted scale factors (not used in baseline)
        c: M × N × L (FP16 output buffer)
    
    Returns:
        c: Output tensor with results
    """
    a, b, sfa, sfb, sfa_permuted, sfb_permuted, c = data
    
    m, k, l = a.shape
    n = b.shape[0]
    
    # Loop over batch dimension (not parallelized - inefficient!)
    for l_idx in range(l):
        # Extract scale factors for this batch element
        # WARNING: to_blocked() does CPU round-trip - major bottleneck!
        scale_a = to_blocked(sfa[:, :, l_idx].cpu()).cuda()
        scale_b = to_blocked(sfb[:, :, l_idx].cpu()).cuda()
        
        # Perform scaled matrix multiplication
        # torch._scaled_mm calls cuBLAS/CUTLASS internally
        result = torch._scaled_mm(
            a[:, :, l_idx],      # M × K (NVFP4)
            b[:, :, l_idx].t(),  # K × N (NVFP4, transposed)
            scale_a=scale_a,     # Blocked scale factors
            scale_b=scale_b,
            out_dtype=torch.float16,
            use_fast_accum=True
        )
        
        # Store result in output buffer
        c[:, :, l_idx] = result
    
    return c


# Ensure function matches reference implementation
custom_kernel = make_match_reference(custom_kernel)
