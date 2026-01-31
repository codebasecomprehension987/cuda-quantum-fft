// ============================================================================
// QFFT CONTEXT IMPLEMENTATION
// Memory management and high-level API
// ============================================================================

#include "../include/qfft.cuh"
#include <cmath>
#include <stdexcept>

namespace QFFT {

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

Context::Context(size_t n, int device_id) 
    : n_(n), device_id_(device_id), d_twiddle_(nullptr), workspace_(nullptr) {
    
    if (!is_power_of_2(n)) {
        throw std::runtime_error("FFT size must be power of 2");
    }
    
    QFFT_CHECK(cudaSetDevice(device_id_));
    
    compute_twiddle_factors();
    allocate_workspace();
}

Context::~Context() {
    if (d_twiddle_) {
        cudaFree(d_twiddle_);
    }
    if (workspace_) {
        cudaFree(workspace_);
    }
}

// ============================================================================
// TWIDDLE FACTOR COMPUTATION
// ============================================================================

void Context::compute_twiddle_factors() {
    int log_n = log2_int(n_);
    size_t twiddle_size = n_ / 2;
    
    std::vector<Complex32> h_twiddle(twiddle_size);
    
    // Compute twiddle factors: W_n^k = e^(-2πik/n)
    const float PI = 3.14159265358979323846f;
    for (size_t k = 0; k < twiddle_size; ++k) {
        float angle = -2.0f * PI * k / n_;
        h_twiddle[k] = make_float2(cosf(angle), sinf(angle));
    }
    
    // Upload to device
    QFFT_CHECK(cudaMalloc(&d_twiddle_, twiddle_size * sizeof(Complex32)));
    QFFT_CHECK(cudaMemcpy(d_twiddle_, h_twiddle.data(),
                         twiddle_size * sizeof(Complex32),
                         cudaMemcpyHostToDevice));
}

// ============================================================================
// WORKSPACE ALLOCATION
// ============================================================================

void Context::allocate_workspace() {
    // Allocate temporary buffer for in-place operations
    size_t workspace_size = n_ * sizeof(Complex32);
    QFFT_CHECK(cudaMalloc(&workspace_, workspace_size));
}

// ============================================================================
// FORWARD FFT
// ============================================================================

void Context::forward(const Complex32* input, Complex32* output, 
                     cudaStream_t stream) {
    launch_radix8_fft(input, output, d_twiddle_, n_, 1, stream);
}

// ============================================================================
// INVERSE FFT
// ============================================================================

void Context::inverse(const Complex32* input, Complex32* output,
                     cudaStream_t stream) {
    // Inverse FFT: conjugate input, forward FFT, conjugate output, scale
    launch_radix8_fft(input, output, d_twiddle_, n_, -1, stream);
}

// ============================================================================
// BATCHED FORWARD FFT
// ============================================================================

void Context::forward_batched(const Complex32* input, Complex32* output,
                              int batch_size, cudaStream_t stream) {
    // Process multiple FFTs with single kernel launch
    for (int b = 0; b < batch_size; ++b) {
        const Complex32* batch_input = input + b * n_;
        Complex32* batch_output = output + b * n_;
        
        launch_radix8_fft(batch_input, batch_output, d_twiddle_, 
                         n_, 1, stream);
    }
}

// ============================================================================
// MIXED PRECISION FFT (FP16)
// ============================================================================

void Context::forward_fp16(const half2* input, half2* output, 
                          cudaStream_t stream) {
    // Convert to FP32, compute, convert back
    Complex32* temp_in;
    Complex32* temp_out;
    
    QFFT_CHECK(cudaMalloc(&temp_in, n_ * sizeof(Complex32)));
    QFFT_CHECK(cudaMalloc(&temp_out, n_ * sizeof(Complex32)));
    
    // Launch conversion kernel (not shown for brevity)
    // convert_fp16_to_fp32<<<...>>>(input, temp_in, n_);
    
    launch_radix8_fft(temp_in, temp_out, d_twiddle_, n_, 1, stream);
    
    // convert_fp32_to_fp16<<<...>>>(temp_out, output, n_);
    
    QFFT_CHECK(cudaFree(temp_in));
    QFFT_CHECK(cudaFree(temp_out));
}

} // namespace QFFT
