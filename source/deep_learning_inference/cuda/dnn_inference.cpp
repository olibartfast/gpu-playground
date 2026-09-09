// Reference implementation of the deep learning inference network.
// This is a naive, unoptimized baseline. For the exercise, implement
// CreateNetworkCandidate in dnn_inference_optimized.cpp with optimized kernels.

#include "dnn_inference.h"
#include "cuda_helpers.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <type_traits>

#include <cooperative_groups.h>

// ---------------------------------------------------------------------------
// NetworkWeights::FromFile
// ---------------------------------------------------------------------------

std::unique_ptr<NetworkWeights> NetworkWeights::FromFile(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());
    assert(std::filesystem::file_size(filename) == sizeof(NetworkWeights));

    auto weights = std::make_unique<NetworkWeights>();
    ifs.read(reinterpret_cast<char*>(weights.get()), sizeof(NetworkWeights));
    assert(ifs.good());
    return weights;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

namespace {

template <typename T>
__host__ __device__ constexpr auto prod(T v) { return v; }

template <typename T, typename... Tail>
__host__ __device__ constexpr auto prod(T v, Tail... tail) {
    return v * prod(tail...);
}

template <typename T, uint32_t n_dimensions>
class TensorView {
public:
    template <typename... Dims>
    constexpr __host__ __device__ TensorView(T* ptr, Dims... dims)
        : dimensions{static_cast<uint32_t>(dims)...}, m_ptr(ptr)
    {
        static_assert(sizeof...(dims) == n_dimensions);
    }

    __host__ __device__ T* ptr() { return m_ptr; }
    __host__ __device__ const T* ptr() const { return m_ptr; }

    template <typename... Idx>
    __host__ __device__ T& at(Idx... idx) {
        static_assert(sizeof...(Idx) == n_dimensions);
        return m_ptr[_calc_flat_index<0, Idx...>(idx...)];
    }

    template <typename... Idx>
    __host__ __device__ const T& at(Idx... idx) const {
        static_assert(sizeof...(Idx) == n_dimensions);
        return m_ptr[_calc_flat_index<0, Idx...>(idx...)];
    }

    const uint32_t dimensions[n_dimensions];

private:
    template <uint32_t dim_idx>
    constexpr __host__ __device__ uint32_t stride_size() const {
        if constexpr (dim_idx < n_dimensions) {
            return dimensions[dim_idx] * stride_size<dim_idx + 1>();
        } else {
            return 1;
        }
    }

    template <uint32_t dim_idx, typename Idx>
    constexpr static __host__ __device__ uint32_t _calc_flat_index(Idx idx) {
        static_assert(dim_idx == n_dimensions - 1);
        return idx;
    }

    template <uint32_t dim_idx, typename Idx, typename... Tail>
    constexpr __host__ __device__ uint32_t _calc_flat_index(Idx idx, Tail... tail) const {
        return idx * stride_size<dim_idx + 1>() + _calc_flat_index<dim_idx + 1, Tail...>(tail...);
    }

    T* m_ptr;
};

template <typename T>
class CudaTensor {
public:
    template <typename... Dims>
    CudaTensor(Dims... dims) : m_size(prod(static_cast<size_t>(dims)...) * sizeof(T)) {
        CUDA_CHECK(cudaMalloc(&m_ptr, m_size));
    }

    T* ptr() { return m_ptr; }
    size_t size() const { return m_size; }

    ~CudaTensor() { CUDA_CHECK(cudaFree(m_ptr)); }

private:
    T* m_ptr;
    size_t m_size;
};

__host__ __device__ inline dim3 operator*(dim3 a, dim3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ inline dim3 operator+(dim3 a, dim3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename F, typename... Args>
void launch_kernel(dim3 total_grid, dim3 block, cudaStream_t stream, F kernel, Args&&... args) {
    assert(total_grid.x > 0 && total_grid.y > 0 && total_grid.z > 0);
    assert(block.x > 0 && block.y > 0 && block.z > 0);
    assert(total_grid.x % block.x == 0);
    assert(total_grid.y % block.y == 0);
    assert(total_grid.z % block.z == 0);

    dim3 grid{total_grid.x / block.x, total_grid.y / block.y, total_grid.z / block.z};
    kernel<<<grid, block, 0, stream>>>(std::forward<Args>(args)...);
}

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Layer 1 — cast uint8 → float. Grid: (H, W, 4).
__global__ void layer1_cast_float(const uint8_t* input_ptr, float* output_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const uint8_t, 3> input(input_ptr, height, width, 4);
    TensorView<float, 3> output(output_ptr, height, width, 4);
    output.at(rank.x, rank.y, rank.z) = static_cast<float>(input.at(rank.x, rank.y, rank.z));
}

// Layer 2 — normalize to [0, 1]. Grid: (H, W, 4).
__global__ void layer2_div_255(const float* input_ptr, float* output_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height, width, 4);
    TensorView<float, 3> output(output_ptr, height, width, 4);
    output.at(rank.x, rank.y, rank.z) = input.at(rank.x, rank.y, rank.z) / 255.0f;
}

// Layer 3 — conv2d 3×3, 4→32 channels, replicate padding. Grid: (H, W, 32).
__global__ void layer3_conv3x3_32(const float* input_ptr, float* output_ptr, const float* layer_weights_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height, width, 4);
    TensorView<float, 3> output(output_ptr, height, width, 32);
    TensorView<const float, 4> weights(layer_weights_ptr, 32, 4, 3, 3);

    const auto f = rank.z;
    float r = 0.f;

    #pragma unroll
    for (unsigned int i = 0; i < 3; i++) {
        #pragma unroll
        for (unsigned int j = 0; j < 3; j++) {
            int x = min((int)height - 1, max(0, (int)rank.x + (int)i - 1));
            int y = min((int)width - 1,  max(0, (int)rank.y + (int)j - 1));
            for (unsigned int c = 0; c < 4; c++) {
                r += input.at(x, y, c) * weights.at(f, c, i, j);
            }
        }
    }
    output.at(rank.x, rank.y, f) = r;
}

// Layers 4, 6, 9, 11 — ReLU. Grid: (H, W, C).
template <unsigned int n_channels>
__global__ void layer_relu(const float* input_ptr, float* output_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height, width, n_channels);
    TensorView<float, 3> output(output_ptr, height, width, n_channels);
    output.at(rank.x, rank.y, rank.z) = max(input.at(rank.x, rank.y, rank.z), 0.f);
}

// Layer 5 — conv1×1, 32→32 channels. Grid: (H, W, 32).
__global__ void layer5_conv1x1_32(const float* input_ptr, float* output_ptr, const float* layer_weights_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height, width, 32);
    TensorView<float, 3> output(output_ptr, height, width, 32);
    TensorView<const float, 4> weights(layer_weights_ptr, 32, 32, 1, 1);

    const auto f = rank.z;
    float r = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        r += input.at(rank.x, rank.y, i) * weights.at(f, i, 0, 0);
    }
    output.at(rank.x, rank.y, f) = r;
}

// Layer 7 — avg pool 2×2 with stride 4. Grid: (H/4, W/4, 32).
__global__ void layer7_avgpool2x2_stride4(const float* input_ptr, float* output_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height * 4, width * 4, 32);
    TensorView<float, 3> output(output_ptr, height, width, 32);

    float r = 0.f;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            r += input.at(rank.x * 4 + i, rank.y * 4 + j, rank.z) * 0.25f;
        }
    }
    output.at(rank.x, rank.y, rank.z) = r;
}

// Layer 8 — exotic conv1×1, 32→256 channels with gamma activation. Grid: (H/4, W/4, 256).
//
// gamma(x, y) = (b*y + 1) * x   if x*y > 0
//               exp(b*y) * x     otherwise
// where b = beta^2
__global__ void layer8_exotic_conv1x1_256(const float* input_ptr, float* output_ptr, const float* layer_weights_ptr, float beta) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height, width, 32);
    TensorView<float, 3> output(output_ptr, height, width, 256);
    TensorView<const float, 4> weights(layer_weights_ptr, 256, 32, 1, 1);

    const auto f = rank.z;
    float r = 0.f;

    constexpr auto gamma = [](float x, float y, float beta) {
        float b = beta * beta;
        return (x * y > 0.f) ? (b * y + 1.f) * x : __expf(b * y) * x;
    };

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        r += gamma(input.at(rank.x, rank.y, i), weights.at(f, i, 0, 0), beta);
    }
    output.at(rank.x, rank.y, f) = r;
}

// Layer 10 — conv1×1, 256→16 channels. Grid: (H/4, W/4, 16).
__global__ void layer10_conv1x1_16(const float* input_ptr, float* output_ptr, const float* layer_weights_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height, width, 256);
    TensorView<float, 3> output(output_ptr, height, width, 16);
    TensorView<const float, 4> weights(layer_weights_ptr, 16, 256, 1, 1);

    const uint32_t f = rank.z;
    float r = 0.f;
    for (unsigned int c = 0; c < 256; c++) {
        r += input.at(rank.x, rank.y, c) * weights.at(f, c, 0, 0);
    }
    output.at(rank.x, rank.y, f) = r;
}

// Layer 12 — clamp to [0, 1]. Grid: (H/4, W/4, 16).
__global__ void layer12_clamp_0_1(const float* input_ptr, float* output_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height, width, 16);
    TensorView<float, 3> output(output_ptr, height, width, 16);
    output.at(rank.x, rank.y, rank.z) = max(0.f, min(1.f, input.at(rank.x, rank.y, rank.z)));
}

// Layer 13 — scale to [0, 255]. Grid: (H/4, W/4, 16).
__global__ void layer13_mult_255(const float* input_ptr, float* output_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height, width, 16);
    TensorView<float, 3> output(output_ptr, height, width, 16);
    output.at(rank.x, rank.y, rank.z) = input.at(rank.x, rank.y, rank.z) * 255.0f;
}

// Layer 14 — cast float → uint8. Grid: (H/4, W/4, 16).
__global__ void layer14_cast_uint8(const float* input_ptr, uint8_t* output_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const float, 3> input(input_ptr, height, width, 16);
    TensorView<uint8_t, 3> output(output_ptr, height, width, 16);
    output.at(rank.x, rank.y, rank.z) = static_cast<uint8_t>(input.at(rank.x, rank.y, rank.z));
}

// Layer 15 — depth-to-space (PixelShuffle ×4): (H/4, W/4, 16) → (H, W). Grid: (H/4, W/4, 16).
__global__ void layer15_depth2space(const uint8_t* input_ptr, uint8_t* output_ptr) {
    const auto thread_block = cg::this_thread_block();
    const auto total_grid = cg::this_grid().dim_blocks() * thread_block.dim_threads();
    const auto height = total_grid.x, width = total_grid.y;
    const auto rank = thread_block.group_index() * thread_block.dim_threads() + thread_block.thread_index();

    TensorView<const uint8_t, 3> input(input_ptr, height, width, 16);
    TensorView<uint8_t, 2> output(output_ptr, 4 * height, 4 * width);
    output.at(rank.x * 4 + rank.z / 4, rank.y * 4 + rank.z % 4) = input.at(rank.x, rank.y, rank.z);
}

// ---------------------------------------------------------------------------
// NetworkImpl — reference (naive) implementation
// ---------------------------------------------------------------------------

class NetworkImpl final : public Network {
public:
    NetworkImpl(size_t height, size_t width,
                const uint8_t* input_ptr, uint8_t* output_ptr,
                cudaStream_t stream)
        : m_height(height), m_width(width),
          m_input_ptr(input_ptr), m_output_ptr(output_ptr), m_stream(stream),
          m_layer1_output(height, width, 4),
          m_layer2_output(height, width, 4),
          m_layer3_output(height, width, 32),
          m_layer4_output(height, width, 32),
          m_layer5_output(height, width, 32),
          m_layer6_output(height, width, 32),
          m_layer7_output(height / 4, width / 4, 32),
          m_layer8_output(height / 4, width / 4, 256),
          m_layer9_output(height / 4, width / 4, 256),
          m_layer10_output(height / 4, width / 4, 16),
          m_layer11_output(height / 4, width / 4, 16),
          m_layer12_output(height / 4, width / 4, 16),
          m_layer13_output(height / 4, width / 4, 16),
          m_layer14_output(height / 4, width / 4, 16),
          m_layer3_weights(32, 4, 3, 3),
          m_layer5_weights(32, 32, 1, 1),
          m_layer8_weights(256, 32, 1, 1),
          m_layer10_weights(16, 256, 1, 1)
    {}

    void Run() override {
        _launch(m_height, m_width, 4, layer1_cast_float, m_input_ptr, m_layer1_output);
        _launch(m_height, m_width, 4, layer2_div_255, m_layer1_output, m_layer2_output);
        _launch(m_height, m_width, 32, layer3_conv3x3_32, m_layer2_output, m_layer3_output, m_layer3_weights);
        _launch(m_height, m_width, 32, layer_relu<32>, m_layer3_output, m_layer4_output);
        _launch(m_height, m_width, 32, layer5_conv1x1_32, m_layer4_output, m_layer5_output, m_layer5_weights);
        _launch(m_height, m_width, 32, layer_relu<32>, m_layer5_output, m_layer6_output);
        _launch(m_height / 4, m_width / 4, 32, layer7_avgpool2x2_stride4, m_layer6_output, m_layer7_output);
        _launch(m_height / 4, m_width / 4, 256, layer8_exotic_conv1x1_256, m_layer7_output, m_layer8_output, m_layer8_weights, m_layer8_beta);
        _launch(m_height / 4, m_width / 4, 256, layer_relu<256>, m_layer8_output, m_layer9_output);
        _launch(m_height / 4, m_width / 4, 16, layer10_conv1x1_16, m_layer9_output, m_layer10_output, m_layer10_weights);
        _launch(m_height / 4, m_width / 4, 16, layer_relu<16>, m_layer10_output, m_layer11_output);
        _launch(m_height / 4, m_width / 4, 16, layer12_clamp_0_1, m_layer11_output, m_layer12_output);
        _launch(m_height / 4, m_width / 4, 16, layer13_mult_255, m_layer12_output, m_layer13_output);
        _launch(m_height / 4, m_width / 4, 16, layer14_cast_uint8, m_layer13_output, m_layer14_output);
        _launch(m_height / 4, m_width / 4, 16, layer15_depth2space, m_layer14_output, m_output_ptr);
    }

    void LoadWeights(const NetworkWeights& weights) override {
        CUDA_CHECK(cudaMemcpy(m_layer3_weights.ptr(), &weights.layer3_weights, sizeof(weights.layer3_weights), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_layer5_weights.ptr(), &weights.layer5_weights, sizeof(weights.layer5_weights), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_layer8_weights.ptr(), &weights.layer8_weights, sizeof(weights.layer8_weights), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_layer10_weights.ptr(), &weights.layer10_weights, sizeof(weights.layer10_weights), cudaMemcpyHostToDevice));
        m_layer8_beta = weights.layer8_beta;
    }

    ~NetworkImpl() override = default;

private:
    template <typename T>
    auto as_arg(T&& arg) {
        using DT = std::decay_t<T>;
        if constexpr (std::is_pointer_v<DT> || std::is_scalar_v<DT>) {
            return std::forward<T>(arg);
        } else {
            return arg.ptr();
        }
    }

    template <typename Kernel, typename... Args>
    void _launch(size_t gx, size_t gy, size_t gz, Kernel kernel, Args&&... args) {
        const dim3 block = {32, 8, 1};
        const dim3 grid = {static_cast<uint32_t>(gx), static_cast<uint32_t>(gy), static_cast<uint32_t>(gz)};
        launch_kernel(grid, block, m_stream, kernel, as_arg(args)...);
    }

    size_t m_height, m_width;
    const uint8_t* m_input_ptr;
    uint8_t* m_output_ptr;
    cudaStream_t m_stream;

    CudaTensor<float>
        m_layer1_output, m_layer2_output,
        m_layer3_output, m_layer4_output,
        m_layer5_output, m_layer6_output,
        m_layer7_output,
        m_layer8_output, m_layer9_output,
        m_layer10_output, m_layer11_output,
        m_layer12_output, m_layer13_output;

    CudaTensor<uint8_t> m_layer14_output;

    float m_layer8_beta = 0.f;
    CudaTensor<float> m_layer3_weights, m_layer5_weights, m_layer8_weights, m_layer10_weights;
};

} // anonymous namespace

std::unique_ptr<Network> CreateNetworkReference(
    size_t image_height, size_t image_width,
    const uint8_t* input_ptr, uint8_t* output_ptr,
    cudaStream_t stream)
{
    return std::make_unique<NetworkImpl>(image_height, image_width, input_ptr, output_ptr, stream);
}
