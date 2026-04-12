# LoRA Linear CUDA Kernel Implementation Plan

## Overview
This document outlines the implementation of a CUDA kernel for LoRA (Low-Rank Adaptation) linear transformation, optimized for performance following the cornerstone principles.

## Reference
- LoRA Paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  - Authors: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
  - Published: June 17, 2021 (v1), revised October 16, 2021 (v2)
  - arXiv ID: 2106.09685

## Mathematical Formulation
```
output = x @ W^T  +  lora_scale * (x @ A^T) @ B^T
```

| Tensor | Shape        | Description           |
|--------|--------------|-----------------------|
| `x`    | `[B, d_in]`  | Input batch           |
| `W`    | `[d_out, d_in]` | Pretrained weights    |
| `A`    | `[rank, d_in]` | LoRA matrix A         |
| `B`    | `[d_out, rank]` | LoRA matrix B         |
| `output`| `[B, d_out]` | Final output          |

## Implementation Approach

### Cornerstone Mapping
The algorithm decomposes into three operations:

| Step | Operation                  | Cornerstone          |
|------|----------------------------|----------------------|
| 1    | `xW = x @ W^T`             | Tiled Partitioning   |
| 2    | `xA = x @ A^T`             | Tiled Partitioning   |
| 3    | `lora = xA @ B^T`          | Tiled Partitioning   |
| 4    | `output = xW + scale * lora`| Map                  |

### Key Components

1. **Tiled GEMM Kernel** (`tiled_gemm_NT`):
   - Implements matrix multiplication for NT layout (no-transpose × transpose)
   - Uses 32x32 tile size for optimal occupancy
   - Includes shared memory padding to avoid bank conflicts
   - Features coalesced memory access patterns
   - Optional warp-level reduction optimization

2. **Fused Elementwise Kernel** (`fused_add_scaled`):
   - Implements `output = base + scale * lora`
   - Pure elementwise map operation
   - No synchronization needed between threads

3. **Stream Parallelism**:
   - Steps 1 and 2 (`xW` and `xA`) run concurrently on separate streams
   - Step 3 (`lora`) depends on completion of step 2
   - Step 4 waits for completion of both steps 1 and 3

### Memory Management
- Intermediate buffers: `xW_buf`, `xA_buf`, `lora_buf`
- Proper allocation and deallocation with CUDA error checking
- Stream creation and destruction for parallel execution

### Error Handling
- Comprehensive CUDA error checking using `CUDA_CHECK` macro
- Validation of all CUDA API calls and kernel launches

## Performance Considerations
- Tile size of 32 matches warp width for optimal memory access patterns
- Stream parallelism hides latency of smaller GEMM operations
- Coalesced global memory access throughout
- Bank conflict avoidance via shared memory padding
- Warp-level reduction options for improved efficiency

## Usage Notes
- Designed for FP32 precision as specified
- Requires CUDA-capable GPU with compute capability suitable for shared memory operations
- Intended as a standalone implementation without external library dependencies