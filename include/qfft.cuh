// ============================================================================
// CUDA QUANTUM FFT LIBRARY
// High-performance FFT with quantum-inspired optimizations
// ============================================================================

#ifndef QFFT_CUH
#define QFFT_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <complex>
#include <vector>
#include <memory>

namespace QFFT {

// Complex number types
using Complex32 = float2;
using Complex64 = double2;

// Error checking macro
#define QFFT_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "QFFT Error: %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Configuration constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;
constexpr int RADIX = 8;

// ============================================================================
// CORE FFT CONTEXT
// ============================================================================

class Context {
public:
    Context(size_t n, int device_id = 0);
    ~Context();

    // Forward/Inverse transforms
    void forward(const Complex32* input, Complex32* output, cudaStream_t stream = 0);
    void inverse(const Complex32* input, Complex32* output, cudaStream_t stream = 0);
    
    // Batched operations
    void forward_batched(const Complex32* input, Complex32* output, 
                        int batch_size, cudaStream_t stream = 0);
    
    // Mixed precision
    void forward_fp16(const half2* input, half2* output, cudaStream_t stream = 0);

    size_t get_size() const { return n_; }
    
private:
    size_t n_;              // Transform size
    int device_id_;
    Complex32* d_twiddle_;  // Twiddle factors
    void* workspace_;       // Temporary workspace
    
    void compute_twiddle_factors();
    void allocate_workspace();
};

// ============================================================================
// KERNEL LAUNCHERS
// ============================================================================

void launch_radix8_fft(const Complex32* input, Complex32* output,
                       const Complex32* twiddle, size_t n, 
                       int direction, cudaStream_t stream);

void launch_stockham_fft(const Complex32* input, Complex32* output,
                         const Complex32* twiddle, size_t n,
                         cudaStream_t stream);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Bit reversal permutation
void bit_reverse_gpu(Complex32* data, size_t n, cudaStream_t stream);

// Power-of-2 check
inline bool is_power_of_2(size_t n) {
    return n && !(n & (n - 1));
}

// Log2 computation
inline int log2_int(size_t n) {
    int result = 0;
    while (n >>= 1) ++result;
    return result;
}

} // namespace QFFT

#endif // QFFT_CUH
