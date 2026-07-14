#include "cuda/dnn_inference.h"
#include "cuda_helpers.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// Image helper
// ---------------------------------------------------------------------------

class Image {
public:
    Image(size_t height, size_t width, size_t n_channels)
        : height(height), width(width), n_channels(n_channels),
          m_data(height * width * n_channels)
    {}

    static Image load(const char* filename, int desired_channels = 0) {
        int w, h, c;
        uint8_t* pixels = stbi_load(filename, &w, &h, &c, desired_channels);
        if (!pixels) {
            std::cerr << "Failed to open image: " << filename << "\n";
            std::terminate();
        }
        if (desired_channels != 0) c = desired_channels;
        Image img(static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(c));
        std::memcpy(img.data(), pixels, img.size());
        stbi_image_free(pixels);
        return img;
    }

    void save_png(const char* filename) const {
        stbi_write_png(filename,
            static_cast<int>(width), static_cast<int>(height),
            static_cast<int>(n_channels), data(),
            static_cast<int>(width * n_channels));
    }

    const size_t height, width, n_channels;
    const uint8_t* data() const { return m_data.data(); }
    uint8_t* data() { return m_data.data(); }
    size_t size() const { return m_data.size(); }
    uint8_t at(size_t i, size_t j, size_t k) const {
        return m_data[i * width * n_channels + j * n_channels + k];
    }

private:
    std::vector<uint8_t> m_data;
};

// ---------------------------------------------------------------------------
// Inference helper
// ---------------------------------------------------------------------------

template <typename CreateFn>
Image run_network(CreateFn create_fn, const Image& input_image) {
    if (input_image.height % 128 != 0 || input_image.width % 128 != 0 || input_image.n_channels != 4) {
        std::cerr << "Invalid image dimensions: need multiples of 128 and 4 channels, "
                  << "got (H=" << input_image.height << ", W=" << input_image.width
                  << ", C=" << input_image.n_channels << ")\n";
        std::terminate();
    }

    Image output(input_image.height, input_image.width, 1);

    void *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_image.size()));
    CUDA_CHECK(cudaMalloc(&d_output, output.size()));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const auto weights = NetworkWeights::FromFile(WEIGHTS_PATH);
    const auto network = create_fn(
        input_image.height, input_image.width,
        static_cast<const uint8_t*>(d_input),
        static_cast<uint8_t*>(d_output),
        stream);

    network->LoadWeights(*weights);

    CUDA_CHECK(cudaMemcpy(d_input, input_image.data(), input_image.size(), cudaMemcpyHostToDevice));
    network->Run();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return output;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

struct Stats { float min, q25, median, q75, max, mean, std; };

static Stats calc_stats(std::vector<float> t) {
    Stats s{};
    if (t.empty()) return s;
    std::sort(t.begin(), t.end());
    auto qi = [&](float q) { return t[std::clamp<size_t>((size_t)(q * t.size()), 0, t.size() - 1)]; };
    s.min    = t.front();
    s.max    = t.back();
    s.q25    = qi(0.25f);
    s.median = qi(0.50f);
    s.q75    = qi(0.75f);
    s.mean   = std::accumulate(t.begin(), t.end(), 0.f) / (float)t.size();
    float var = std::transform_reduce(t.begin(), t.end(), 0.f, std::plus<>(),
        [m = s.mean](float x) { return (x - m) * (x - m); }) / (float)t.size();
    s.std = sqrtf(var);
    return s;
}

static void print_stats_row(const char* name, const Stats& s) {
    auto col = [](float v) {
        std::cout << ' ' << std::fixed << std::setw(9) << std::setfill(' ')
                  << std::setprecision(5) << std::right << v << " |";
    };
    std::cout << std::setw(9) << std::setfill(' ') << std::left << name << " |";
    col(s.min); col(s.q25); col(s.median); col(s.q75); col(s.max); col(s.mean); col(s.std);
    std::cout << "\n";
}

void benchmark() {
#ifndef NDEBUG
    std::cerr << "WARNING: debug build — benchmarks should be run in release mode\n";
#endif

    const std::vector<std::pair<uint32_t, uint32_t>> sizes = {
        {512, 512}, {256, 256}, {256, 512}, {512, 384}, {768, 128}
    };

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr size_t N = 100;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> events(N);
    for (size_t i = 0; i < N; i++) {
        CUDA_CHECK(cudaEventCreate(&events[i].first));
        CUDA_CHECK(cudaEventCreate(&events[i].second));
    }

    const auto weights = std::make_unique<NetworkWeights>();

    for (const auto& [H, W] : sizes) {
        void *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  (size_t)H * W * 4));
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)H * W));

        auto time_network = [&](Network* net) {
            std::vector<float> timings(N);
            net->LoadWeights(*weights);
            for (size_t i = 0; i < N; i++) {
                CUDA_CHECK(cudaEventRecord(events[i].first, stream));
                net->Run();
                CUDA_CHECK(cudaEventRecord(events[i].second, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (size_t i = 0; i < N; i++) {
                CUDA_CHECK(cudaEventElapsedTime(&timings[i], events[i].first, events[i].second));
            }
            return timings;
        };

        const auto ref = CreateNetworkReference(H, W, (uint8_t*)d_in, (uint8_t*)d_out, stream);
        const auto cand = CreateNetworkCandidate(H, W, (uint8_t*)d_in, (uint8_t*)d_out, stream);

        std::cout << "Image " << H << "x" << W << " — " << N << " runs (ms)\n";
        std::cout << "          |    Min    |    Q25    |    Med    |    Q75    |    Max    |    Mean   |    Std    |\n";
        std::cout << "---------------------------------------------------------------------------------------------------\n";

        const Stats ref_s  = calc_stats(time_network(ref.get()));
        const Stats cand_s = calc_stats(time_network(cand.get()));
        print_stats_row("reference", ref_s);
        print_stats_row("candidate", cand_s);
        std::cout << "Speedup: " << std::setprecision(2) << ref_s.mean / cand_s.mean << "x\n\n";

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    for (size_t i = 0; i < N; i++) {
        CUDA_CHECK(cudaEventDestroy(events[i].first));
        CUDA_CHECK(cudaEventDestroy(events[i].second));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------

bool check_correctness(const char* input_filename) {
    constexpr int threshold = 3;
    const Image input = Image::load(input_filename, 4);

    Image ref_out  = run_network(CreateNetworkReference, input);
    Image cand_out = run_network(CreateNetworkCandidate, input);

    unsigned n_diff = 0;
    for (size_t i = 0; i < ref_out.height && n_diff < 10; i++) {
        for (size_t j = 0; j < ref_out.width && n_diff < 10; j++) {
            int vr = ref_out.at(i, j, 0), vc = cand_out.at(i, j, 0);
            if (abs(vr - vc) > threshold) {
                std::cerr << "Pixel [" << i << "," << j << "]: expected " << vr << " got " << vc << "\n";
                n_diff++;
            }
        }
    }
    return n_diff == 0;
}

// ---------------------------------------------------------------------------
// image_infer mode
// ---------------------------------------------------------------------------

void image_infer(const char* in_path, const char* out_path) {
    const Image input = Image::load(in_path, 4);
    Image output = run_network(CreateNetworkCandidate, input);
    output.save_png(out_path);
    std::cout << "Saved: " << out_path << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

[[noreturn]] static void usage() {
    std::cerr << "Usage:\n"
              << "  deep_learning_inference benchmark\n"
              << "  deep_learning_inference image_infer <input.png> <output.png>\n"
              << "  deep_learning_inference correctness <input.png>\n";
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2) usage();
    const std::string_view mode = argv[1];

    if (mode == "benchmark") {
        benchmark();
    } else if (mode == "image_infer") {
        if (argc < 4) usage();
        image_infer(argv[2], argv[3]);
    } else if (mode == "correctness") {
        if (argc < 3) usage();
        if (!check_correctness(argv[2])) {
            std::cerr << "FAILED — candidate output differs from reference\n";
            return 1;
        }
        std::cout << "PASSED — candidate matches reference\n";
    } else {
        usage();
    }

    return 0;
}
