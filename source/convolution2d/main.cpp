#ifdef GPU_OPENCL_CPP_BACKEND
#include "opencl_cpp/convolution2d.h"
#elif defined(GPU_OPENCL_BACKEND)
#include "opencl/convolution2d.h"
#else
#include "cuda/convolution2d.h"
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void print_matrix(const float* data, int rows, int cols, const char* name) {
    std::printf("%s:\n", name);
    for (int r = 0; r < rows; r++) {
        std::printf("  ");
        for (int c = 0; c < cols; c++) {
            std::printf("%7.2f", data[r * cols + c]);
        }
        std::printf("\n");
    }
    std::printf("\n");
}

static bool compare_results(const float* expected, const float* actual, int count, float tolerance) {
    float max_diff = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < count; i++) {
        float diff = std::fabs(expected[i] - actual[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            if (mismatches < 5) {
                std::printf("Mismatch at %d: cpu=%f gpu=%f diff=%f\n",
                            i, expected[i], actual[i], diff);
            }
            mismatches++;
        }
    }
    std::printf("Max abs diff: %e\n", max_diff);
    if (mismatches > 5) std::printf("... plus %d more mismatches\n", mismatches - 5);
    return mismatches == 0;
}

int main() {
    std::printf("=== 2D Convolution Test ===\n\n");

    const int input_rows = 5;
    const int input_cols = 5;
    const int kernel_rows = 3;
    const int kernel_cols = 3;
    const int out_rows = input_rows - kernel_rows + 1;
    const int out_cols = input_cols - kernel_cols + 1;

    float input[input_rows * input_cols] = {
        1,  2,  3,  4,  5,
        6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    float kernel[kernel_rows * kernel_cols] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };
    float output_cpu[out_rows * out_cols] = {};
    float output_gpu[out_rows * out_cols] = {};

    print_matrix(input, input_rows, input_cols, "Input");
    print_matrix(kernel, kernel_rows, kernel_cols, "Kernel");

    auto start = std::chrono::steady_clock::now();
    convolution2d_cpu(input, kernel, output_cpu, input_rows, input_cols, kernel_rows, kernel_cols);
    auto end = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::steady_clock::now();
    convolution2d_gpu(input, kernel, output_gpu, input_rows, input_cols, kernel_rows, kernel_cols);
    end = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(end - start).count();

    print_matrix(output_cpu, out_rows, out_cols, "CPU Output");
    print_matrix(output_gpu, out_rows, out_cols, "GPU Output");
    std::printf("Expected first test output should be all -6.00\n");
    bool small_ok = compare_results(output_cpu, output_gpu, out_rows * out_cols, 1e-4f);
    std::printf("CPU time: %.3f ms\nGPU time: %.3f ms\n\n", cpu_ms, gpu_ms);

    const int large_input_rows = 32;
    const int large_input_cols = 32;
    const int large_kernel_rows = 5;
    const int large_kernel_cols = 5;
    const int large_out_rows = large_input_rows - large_kernel_rows + 1;
    const int large_out_cols = large_input_cols - large_kernel_cols + 1;

    float* large_input = new float[large_input_rows * large_input_cols];
    float* large_kernel = new float[large_kernel_rows * large_kernel_cols];
    float* large_cpu = new float[large_out_rows * large_out_cols];
    float* large_gpu = new float[large_out_rows * large_out_cols];

    std::srand(42);
    for (int i = 0; i < large_input_rows * large_input_cols; i++) {
        large_input[i] = (float)(std::rand() % 21 - 10) / 5.0f;
    }
    for (int i = 0; i < large_kernel_rows * large_kernel_cols; i++) {
        large_kernel[i] = (float)(std::rand() % 11 - 5) / 5.0f;
    }

    convolution2d_cpu(large_input, large_kernel, large_cpu,
                      large_input_rows, large_input_cols,
                      large_kernel_rows, large_kernel_cols);
    convolution2d_gpu(large_input, large_kernel, large_gpu,
                      large_input_rows, large_input_cols,
                      large_kernel_rows, large_kernel_cols);

    bool large_ok = compare_results(large_cpu, large_gpu, large_out_rows * large_out_cols, 1e-4f);
    std::printf("\nLarge test: %s\n", large_ok ? "PASSED" : "FAILED");

    delete[] large_input;
    delete[] large_kernel;
    delete[] large_cpu;
    delete[] large_gpu;

    bool ok = small_ok && large_ok;
    std::printf("\nOverall result: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
