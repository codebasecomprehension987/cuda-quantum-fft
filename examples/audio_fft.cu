// ============================================================================
// EXAMPLE: Audio Signal Processing with QFFT
// Demonstrates FFT usage for spectral analysis
// ============================================================================

#include "../include/qfft.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace QFFT;

// ============================================================================
// HELPER: Generate test signal
// ============================================================================

void generate_signal(std::vector<Complex32>& signal, size_t n) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    const float PI = 3.14159265358979323846f;
    
    for (size_t i = 0; i < n; ++i) {
        // Composite signal: 440 Hz + 880 Hz (A4 + A5 notes)
        float t = static_cast<float>(i) / n;
        float sample = sinf(2 * PI * 440 * t) + 0.5f * sinf(2 * PI * 880 * t);
        
        // Add noise
        sample += 0.1f * dist(gen);
        
        signal[i] = make_float2(sample, 0.0f);
    }
}

// ============================================================================
// HELPER: Compute magnitude spectrum
// ============================================================================

void compute_magnitude(const std::vector<Complex32>& spectrum, 
                      std::vector<float>& magnitude) {
    for (size_t i = 0; i < spectrum.size(); ++i) {
        magnitude[i] = sqrtf(spectrum[i].x * spectrum[i].x + 
                            spectrum[i].y * spectrum[i].y);
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    const size_t N = 1024 * 1024;  // 1M sample FFT
    
    std::cout << "=== CUDA Quantum FFT Demo ===" << std::endl;
    std::cout << "Transform size: " << N << std::endl;
    
    // Generate input signal
    std::vector<Complex32> h_signal(N);
    generate_signal(h_signal, N);
    
    // Allocate device memory
    Complex32 *d_input, *d_output;
    QFFT_CHECK(cudaMalloc(&d_input, N * sizeof(Complex32)));
    QFFT_CHECK(cudaMalloc(&d_output, N * sizeof(Complex32)));
    
    // Copy to device
    QFFT_CHECK(cudaMemcpy(d_input, h_signal.data(), 
                         N * sizeof(Complex32),
                         cudaMemcpyHostToDevice));
    
    // Create FFT context
    std::cout << "Initializing QFFT context..." << std::endl;
    Context fft_ctx(N);
    
    // Warm-up run
    fft_ctx.forward(d_input, d_output);
    QFFT_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int NUM_RUNS = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_RUNS; ++i) {
        fft_ctx.forward(d_input, d_output);
    }
    QFFT_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time_ms = duration.count() / (1000.0 * NUM_RUNS);
    double gflops = (5.0 * N * log2(N)) / (avg_time_ms * 1e6);  // FFT complexity
    
    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Bandwidth: " << (2 * N * sizeof(Complex32) / avg_time_ms / 1e6) 
              << " GB/s" << std::endl;
    
    // Copy result back
    std::vector<Complex32> h_spectrum(N);
    QFFT_CHECK(cudaMemcpy(h_spectrum.data(), d_output,
                         N * sizeof(Complex32),
                         cudaMemcpyDeviceToHost));
    
    // Compute magnitude spectrum
    std::vector<float> magnitude(N);
    compute_magnitude(h_spectrum, magnitude);
    
    // Find peaks
    std::cout << "\n=== Top 5 Frequency Components ===" << std::endl;
    for (int peak = 0; peak < 5; ++peak) {
        auto max_it = std::max_element(magnitude.begin(), magnitude.end());
        size_t max_idx = std::distance(magnitude.begin(), max_it);
        float freq = static_cast<float>(max_idx) / N;
        
        std::cout << "Peak " << peak + 1 << ": " 
                  << freq * 44100 << " Hz (magnitude: " 
                  << *max_it << ")" << std::endl;
        
        *max_it = 0;  // Remove this peak
    }
    
    // Test inverse FFT
    std::cout << "\n=== Testing Inverse FFT ===" << std::endl;
    Complex32* d_reconstructed;
    QFFT_CHECK(cudaMalloc(&d_reconstructed, N * sizeof(Complex32)));
    
    fft_ctx.inverse(d_output, d_reconstructed);
    
    std::vector<Complex32> h_reconstructed(N);
    QFFT_CHECK(cudaMemcpy(h_reconstructed.data(), d_reconstructed,
                         N * sizeof(Complex32),
                         cudaMemcpyDeviceToHost));
    
    // Compute reconstruction error
    float max_error = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float err = fabsf(h_signal[i].x - h_reconstructed[i].x);
        max_error = fmaxf(max_error, err);
    }
    
    std::cout << "Max reconstruction error: " << max_error << std::endl;
    std::cout << "Reconstruction quality: " 
              << (max_error < 1e-4f ? "EXCELLENT" : "CHECK IMPLEMENTATION") 
              << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_reconstructed);
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    
    return 0;
}
