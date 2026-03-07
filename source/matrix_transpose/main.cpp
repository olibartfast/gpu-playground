#ifdef GPU_OPENCL_BACKEND
#include "opencl/matrix_transpose.h"
#else
#include "cuda/matrix_transpose.h"
#endif
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>

#define PRINT

void print_matrix(const float* matrix, int rows, int cols, const std::string& message = "") {
    if (!message.empty()) std::cout << message << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) std::cout << matrix[i * cols + j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int rows = 4;
    int cols = 6;
    int input_size = rows * cols;
    int output_size = cols * rows;

    float* input = (float*)malloc(sizeof(float) * input_size);
    float* output_cpu = (float*)malloc(sizeof(float) * output_size);
    float* output_gpu = (float*)malloc(sizeof(float) * output_size);

    if (!input || !output_cpu || !output_gpu) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return 1;
    }

    for (int i = 0; i < input_size; i++) input[i] = (float)(i + 1);

    #ifdef PRINT
    print_matrix(input, rows, cols, "Original matrix");
    #endif

    auto start = std::chrono::steady_clock::now();
    matrix_transpose_cpu(input, output_cpu, rows, cols);
    auto end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print_matrix(output_cpu, cols, rows, "CPU transposed");
    #endif
    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();
    matrix_transpose_gpu(input, output_gpu, rows, cols);
    end = std::chrono::steady_clock::now();
    #ifdef PRINT
    print_matrix(output_gpu, cols, rows, "GPU transposed");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    bool results_match = true;
    for (int i = 0; i < output_size; i++) {
        if (output_cpu[i] != output_gpu[i]) { results_match = false; break; }
    }
    std::cout << (results_match ? "Results match!" : "Results do not match!") << std::endl;

    free(input);
    free(output_cpu);
    free(output_gpu);
    return results_match ? 0 : 1;
}
