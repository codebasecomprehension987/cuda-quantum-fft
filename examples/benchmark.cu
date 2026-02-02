// ============================================================================
// FFT BENCHMARK SUITE
// Performance testing across different sizes and configurations
// ============================================================================

#include "../include/qfft.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace QFFT;

struct BenchmarkResult {
    size_t n;
    double time_ms;
    double gflops;
    double bandwidth_gb;
};

// ============================================================================
// BENCHMARK SINGLE SIZE
// ============================================================================

BenchmarkResult benchmark_size(size_t n, int num_runs = 100) {
    // Allocate
    Complex32 *d_input, *d_output;
    QFFT_CHECK(cudaMalloc(&d_input, n * sizeof(Complex32)));
    QFFT_CHECK(cudaMalloc(&d_output, n * sizeof(Complex32)));
    
    // Initialize with random data
    std::vector<Complex32> h_data(n);
    for (size_t i = 0; i < n; ++i) {
        h_data[i] = make_float2(
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX
        );
    }
    QFFT_CHECK(cudaMemcpy(d_input, h_data.data(), 
                         n * sizeof(Complex32),
                         cudaMemcpyHostToDevice));
    
    // Create context
    Context ctx(n);
    
    // Warm-up
    for (int i = 0; i < 5; ++i) {
        ctx.forward(d_input, d_output);
    }
    QFFT_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        ctx.forward(d_input, d_output);
    }
    QFFT_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double time_ms = duration.count() / (1000.0 * num_runs);
    
    // Calculate metrics
    double log_n = log2(static_cast<double>(n));
    double flops = 5.0 * n * log_n;  // FFT complexity
    double gflops = flops / (time_ms * 1e6);
    
    double bytes = 2 * n * sizeof(Complex32);  // Read + write
    double bandwidth_gb = bytes / (time_ms * 1e6);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return {n, time_ms, gflops, bandwidth_gb};
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

int main() {
    std::cout << "=== CUDA Quantum FFT Benchmark Suite ===" << std::endl;
    
    // Get device properties
    cudaDeviceProp prop;
    QFFT_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory Clock: " << prop.memoryClockRate / 1e6 << " GHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    
    double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    std::cout << "Peak Memory Bandwidth: " << peak_bandwidth << " GB/s" << std::endl;
    
    // Benchmark different sizes
    std::vector<size_t> sizes = {
        1024,           // 1K
        4096,           // 4K
        16384,          // 16K
        65536,          // 64K
        262144,         // 256K
        1048576,        // 1M
        4194304,        // 4M
        16777216        // 16M
    };
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << std::left << std::setw(12) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(20) << "Bandwidth (GB/s)"
              << std::setw(18) << "BW Efficiency (%)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    for (size_t n : sizes) {
        auto result = benchmark_size(n, 50);
        
        double efficiency = (result.bandwidth_gb / peak_bandwidth) * 100.0;
        
        std::cout << std::left << std::setw(12) << result.n
                  << std::fixed << std::setprecision(3)
                  << std::setw(15) << result.time_ms
                  << std::setw(15) << result.gflops
                  << std::setw(20) << result.bandwidth_gb
                  << std::setw(18) << efficiency << std::endl;
    }
    
    std::cout << std::string(80, '=') << std::endl;
    
    // Batched benchmark
    std::cout << "\n=== Batched FFT Performance ===" << std::endl;
    
    size_t batch_size = 16;
    size_t fft_size = 65536;
    
    Complex32 *d_input, *d_output;
    QFFT_CHECK(cudaMalloc(&d_input, batch_size * fft_size * sizeof(Complex32)));
    QFFT_CHECK(cudaMalloc(&d_output, batch_size * fft_size * sizeof(Complex32)));
    
    Context ctx(fft_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    ctx.forward_batched(d_input, d_output, batch_size);
    QFFT_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double batch_time_ms = duration.count() / 1000.0;
    
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "FFT size: " << fft_size << std::endl;
    std::cout << "Total time: " << batch_time_ms << " ms" << std::endl;
    std::cout << "Time per FFT: " << batch_time_ms / batch_size << " ms" << std::endl;
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
