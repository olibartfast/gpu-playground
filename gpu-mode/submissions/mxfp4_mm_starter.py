"""
Starter submission for the current AMD qualifier `mxfp4-mm`.

This is the easiest kernel to begin with because it is just:
1. Quantize A to MXFP4 with block-32 scaling.
2. Run the aiter a4w4 GEMM against the pre-shuffled B weights.

Reference source:
  problems/amd_202602/mxfp4-mm/submission.py in gpu-mode/reference-kernels
"""

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    def quantize_a_to_mxfp4(x):
        x_fp4, scales = dynamic_mxfp4_quant(x)
        return x_fp4.view(dtypes.fp4x2), e8m0_shuffle(scales).view(dtypes.fp8_e8m0)

    a, b, b_q, b_shuffle, b_scale_sh = data
    a = a.contiguous()

    a_q, a_scale_sh = quantize_a_to_mxfp4(a)

    return aiter.gemm_a4w4(
        a_q,
        b_shuffle,
        a_scale_sh,
        b_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
