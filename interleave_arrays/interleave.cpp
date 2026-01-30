// nvcc -x cu -arch=compute_70 -code=sm_70,compute_70 -std=c++17 -o interleave interleave.cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <vector>

#define PRINT 
#define BSIZE 256

void interleave(const float* d_A, const float* d_B, float* d_output, int N);

void print(const float* input, int N, const std::string& message = "") {
    if (!message.empty()) {
        std::cout << message << ": ";
    }
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < N; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Each thread handles one element from A and one from B
        // Interleaving pattern: [A[0], B[0], A[1], B[1], A[2], B[2], ...]
        output[2 * idx] = A[idx];
        output[2 * idx + 1] = B[idx];
    }
}

void interleave_cpu(const float* A, const float* B, float* output, int N) {
    // Interleave arrays A and B element by element
    for (int i = 0; i < N; i++) {
        output[2 * i] = A[i];
        output[2 * i + 1] = B[i];
    }
}

void interleave(const float* d_A, const float* d_B, float* d_output, int N) {
    int threadsPerBlock = BSIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_output, N);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Use N=8 for printing and correctness check.
    // For performance, use a much larger N and comment out the #define PRINT.
    int N = 8;
    // int N = 1 << 20; // 1,048,576 elements for performance test

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> output_cpu(2 * N);
    std::vector<float> output_gpu(2 * N);

    // Initialize arrays with some sample values
    if (N == 8) {
        float sample_A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float sample_B[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
        for (int i = 0; i < N; i++) {
            A[i] = sample_A[i];
            B[i] = sample_B[i];
        }
    } else {
        // For large N, fill with random-like data
        for (int i = 0; i < N; i++) {
            A[i] = (float)(i % 100);
            B[i] = (float)((i % 100) + 100);
        }
    }
    
    #ifdef PRINT
    print(A.data(), N, "Array A");
    print(B.data(), N, "Array B");
    #endif

    // --- CPU Interleave ---
    auto start_cpu = std::chrono::steady_clock::now();
    interleave_cpu(A.data(), B.data(), output_cpu.data(), N);
    auto end_cpu = std::chrono::steady_clock::now();
    
    #ifdef PRINT
    print(output_cpu.data(), 2 * N, "CPU Interleave");
    #endif

    std::cout << "CPU time: " << std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count() << " ms" << std::endl << std::endl;
    
    // --- GPU Interleave ---
    float *d_A, *d_B, *d_output;
    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * 2 * N);  // Output is twice the size

    auto start_gpu = std::chrono::steady_clock::now();
    
    cudaMemcpy(d_A, A.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    
    // Call interleave kernel
    interleave(d_A, d_B, d_output, N);
    
    // Ensure all GPU work is done before stopping the timer
    cudaDeviceSynchronize();

    cudaMemcpy(output_gpu.data(), d_output, sizeof(float) * 2 * N, cudaMemcpyDeviceToHost);
    
    auto end_gpu = std::chrono::steady_clock::now();

    #ifdef PRINT
    print(output_gpu.data(), 2 * N, "GPU Interleave");
    #endif
    std::cout << "GPU time: " << std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count() << " ms" << std::endl << std::endl;

    // --- Verification ---
    bool results_match = true;
    const float tolerance = 1e-5f;
    for (int i = 0; i < 2 * N; i++) {
        if (fabsf(output_cpu[i] - output_gpu[i]) > tolerance) {
            results_match = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << output_cpu[i] 
                      << " GPU=" << output_gpu[i] << std::endl;
            break;
        }
    }
    
    if (results_match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);
    return 0;
}
