// Candidate implementation.
//
// WHAT THE NETWORK DOES
// ----------------------
// The network is a small CNN that converts RGBA uint8 images to grayscale.
// It is intentionally designed to exercise a range of GPU inference patterns:
//
//  - A 3×3 depthwise-style convolution (layer 3): tests memory coalescing
//    and shared-memory tiling for stencil access patterns.
//  - Two 1×1 convolutions (layers 5 and 10): pure channel mixing — ideal for
//    GEMM-based or tensor-core implementations.
//  - An "exotic" 1×1 conv (layer 8) with a gated non-linearity (gamma):
//    each input-weight pair goes through one of two activation paths before
//    accumulation, making it harder to vectorize naively.
//  - Average pooling with stride 4 (layer 7): spatial downsampling.
//  - PixelShuffle ×4 (layer 15): rearranges 16 feature-map channels into a
//    full-resolution grayscale output — a pure scatter/gather.
//
// OPTIMIZATION IDEAS
// -------------------
//  1. Fuse elementwise layers (cast + div255, ReLU, clamp, mul255, cast) to
//     eliminate redundant global memory round-trips.
//  2. Use shared-memory tiling for the 3×3 conv (layer 3).
//  3. Use wmma / tensor cores for the large 1×1 convs (layers 5, 8, 10).
//  4. Fuse AvgPool + ExoticConv + ReLU (layers 7–9) into one kernel.
//  5. Use CUDA graphs to amortize per-launch overhead across many small kernels.
//  6. Pre-transform weights in LoadWeights() (e.g., transpose for better
//     memory access patterns) — LoadWeights is not counted in benchmark time.
//
// BASELINE
// --------
// The candidate delegates to the reference network so the exercise starts from
// a correctness-safe implementation. Replace this delegation incrementally as
// optimizations are introduced and validate each change against the reference.
//
// HOW TO TEST
// ------------
//  Generate weights first: python generate_weights.py
//
//  ./deep_learning_inference benchmark
//  ./deep_learning_inference correctness images/image0.png
//  ./deep_learning_inference image_infer  images/image0.png out.png
//
// See dnn_inference.h for the full architecture description and gamma definition.

#include "dnn_inference.h"
#include "cuda_helpers.h"

namespace {

class CandidateNetworkImpl final : public Network {
public:
    CandidateNetworkImpl(size_t height, size_t width,
                         const uint8_t* input_ptr, uint8_t* output_ptr,
                         cudaStream_t stream)
        : m_baseline(CreateNetworkReference(
              height, width, input_ptr, output_ptr, stream)) {}

    void Run() override {
        m_baseline->Run();
    }

    void LoadWeights(const NetworkWeights& weights) override {
        m_baseline->LoadWeights(weights);
    }

    ~CandidateNetworkImpl() override = default;

private:
    std::unique_ptr<Network> m_baseline;
};

} // anonymous namespace

std::unique_ptr<Network> CreateNetworkCandidate(
    size_t image_height, size_t image_width,
    const uint8_t* input_ptr, uint8_t* output_ptr,
    cudaStream_t stream)
{
    return std::make_unique<CandidateNetworkImpl>(image_height, image_width, input_ptr, output_ptr, stream);
}
