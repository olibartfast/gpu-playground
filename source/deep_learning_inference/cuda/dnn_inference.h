#pragma once
//
// Deep Learning Inference — CUDA exercise
// ========================================
//
// The network is a small feed-forward CNN that converts RGBA images to
// grayscale.  It was designed to showcase a variety of GPU inference tricks
// rather than to produce visually meaningful results.
//
// Architecture (layer indices match the kernel names in inference.cpp):
//
//  Input: uint8 RGBA image in HWC layout, values 0–255.
//
//  1. Cast uint8 → float32
//  2. Divide by 255                        → normalized [0, 1] in HWC
//  3. Conv2d(4→32, 3×3, replicate pad)     → spatial feature extraction
//  4. ReLU
//  5. Conv2d(32→32, 1×1)                  → channel mixing
//  6. ReLU
//  7. AvgPool2d(kernel=2, stride=4)        → downsample 4× (H/4, W/4)
//  8. ExoticConv1x1(32→256)               → learned gamma activation (see below)
//  9. ReLU
// 10. Conv2d(256→16, 1×1)                 → channel reduction
// 11. ReLU
// 12. Clamp [0, 1]
// 13. × 255
// 14. Cast float32 → uint8
// 15. PixelShuffle(×4)                    → (H/4, W/4, 16) → (H, W, 1)
//
//  Output: uint8 grayscale image in HW layout.
//
// The "ExoticConv1x1" in layer 8 is a 1×1 convolution where each input-weight
// product is passed through a gated non-linearity (gamma) before accumulation:
//
//   output[o] = Σ_c  gamma( input[c], weight[o,c] )
//
//   gamma(x, y) = (β²·y + 1) · x    if x·y > 0   (both same sign → positive)
//                 exp(β²·y) · x      otherwise     (sign mismatch → exp gate)
//
// This makes the effective weight depend on the sign of the input, giving the
// layer both linear and exponential behaviour within a single convolution.
//
// I/O constraints (from the original exercise):
//   - Input/output GPU pointers are 256-byte aligned.
//   - Image dimensions are always multiples of 128.
//
// Files:
//   cuda/dnn_inference.cpp           — naive reference implementation
//   cuda/dnn_inference_optimized.cpp — TODO: your optimized implementation
//   generate_weights.py              — trains the network and writes weights.bin
//   network_definition.py            — PyTorch definition (ground truth)

#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime.h>

struct NetworkWeights {
    float layer3_weights[32][4][3][3];
    float layer5_weights[32][32];
    float layer8_weights[256][32];
    float layer8_beta;
    float layer10_weights[16][256];

    static std::unique_ptr<NetworkWeights> FromFile(const std::string& filename);
};

class Network {
public:
    virtual void LoadWeights(const NetworkWeights& weights) = 0;
    virtual void Run() = 0;
    virtual ~Network() = default;
};

// Reference implementation (naive, unoptimized)
std::unique_ptr<Network> CreateNetworkReference(
    size_t image_height, size_t image_width,
    const uint8_t* input_ptr, uint8_t* output_ptr,
    cudaStream_t stream);

// Optimized candidate — implement this
std::unique_ptr<Network> CreateNetworkCandidate(
    size_t image_height, size_t image_width,
    const uint8_t* input_ptr, uint8_t* output_ptr,
    cudaStream_t stream);
