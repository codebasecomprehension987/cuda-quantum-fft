// ============================================================================
// RADIX-8 FFT KERNEL
// Optimized Cooley-Tukey butterfly with shared memory
// ============================================================================

#include "../include/qfft.cuh"
#include <cmath>

namespace QFFT {

// ============================================================================
// DEVICE FUNCTIONS - Complex arithmetic
// ============================================================================

__device__ __forceinline__ 
Complex32 complex_mul(Complex32 a, Complex32 b) {
    return make_float2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__device__ __forceinline__
Complex32 complex_add(Complex32 a, Complex32 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__
Complex32 complex_sub(Complex32 a, Complex32 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__
Complex32 complex_conj(Complex32 a) {
    return make_float2(a.x, -a.y);
}

// ============================================================================
// RADIX-8 BUTTERFLY OPERATION
// ============================================================================

__device__ void radix8_butterfly(Complex32* shared_data, int idx, 
                                 const Complex32* twiddle, int stage) {
    const float SQRT2_2 = 0.70710678118f;
    
    // Load 8 elements
    Complex32 x[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = shared_data[idx + i * blockDim.x];
    }
    
    // Stage 1: 4 radix-2 butterflies
    Complex32 t0 = complex_add(x[0], x[4]);
    Complex32 t1 = complex_sub(x[0], x[4]);
    Complex32 t2 = complex_add(x[2], x[6]);
    Complex32 t3 = complex_sub(x[2], x[6]);
    Complex32 t4 = complex_add(x[1], x[5]);
    Complex32 t5 = complex_sub(x[1], x[5]);
    Complex32 t6 = complex_add(x[3], x[7]);
    Complex32 t7 = complex_sub(x[3], x[7]);
    
    // Stage 2: Apply twiddle factors and combine
    Complex32 w = twiddle[idx * stage];
    
    x[0] = complex_add(t0, t2);
    x[1] = complex_mul(complex_add(t0, complex_mul(t2, make_float2(-1, 0))), w);
    x[2] = complex_add(t4, t6);
    x[3] = complex_mul(complex_add(t4, complex_mul(t6, make_float2(-1, 0))), w);
    
    // Apply rotations for other outputs
    Complex32 rot45 = make_float2(SQRT2_2, -SQRT2_2);
    x[4] = complex_mul(complex_add(t1, complex_mul(t3, make_float2(0, -1))), w);
    x[5] = complex_mul(complex_mul(complex_sub(t1, complex_mul(t3, make_float2(0, -1))), rot45), w);
    x[6] = complex_mul(complex_add(t5, complex_mul(t7, make_float2(0, -1))), w);
    x[7] = complex_mul(complex_mul(complex_sub(t5, complex_mul(t7, make_float2(0, -1))), rot45), w);
    
    // Store back
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        shared_data[idx + i * blockDim.x] = x[i];
    }
}

// ============================================================================
// MAIN RADIX-8 FFT KERNEL
// ============================================================================

__global__ void radix8_fft_kernel(const Complex32* __restrict__ input,
                                  Complex32* __restrict__ output,
                                  const Complex32* __restrict__ twiddle,
                                  size_t n, int direction) {
    extern __shared__ Complex32 shared[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    
    // Load data to shared memory with coalescing
    int stride = gridDim.x * blockDim.x;
    #pragma unroll 4
    for (int i = gid; i < n; i += stride) {
        shared[tid + (i / blockDim.x) * blockDim.x] = input[i];
    }
    __syncthreads();
    
    // Compute log2(n) for number of stages
    int stages = 0;
    size_t temp = n;
    while (temp > 1) {
        temp >>= 3;  // Divide by 8
        stages++;
    }
    
    // Perform radix-8 FFT stages
    for (int stage = 0; stage < stages; ++stage) {
        int step = 1 << (3 * stage);  // 8^stage
        
        if (tid < (n / 8)) {
            radix8_butterfly(shared, tid, twiddle, step);
        }
        __syncthreads();
    }
    
    // Write back with bit-reversal for natural order output
    #pragma unroll 4
    for (int i = gid; i < n; i += stride) {
        Complex32 val = shared[tid + (i / blockDim.x) * blockDim.x];
        
        // Apply normalization for inverse FFT
        if (direction < 0) {
            val.x /= n;
            val.y /= n;
        }
        
        output[i] = val;
    }
}

// ============================================================================
// KERNEL LAUNCHER
// ============================================================================

void launch_radix8_fft(const Complex32* input, Complex32* output,
                       const Complex32* twiddle, size_t n,
                       int direction, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Limit blocks for occupancy
    blocks = min(blocks, 2048);
    
    size_t shared_mem = threads * 8 * sizeof(Complex32);
    
    radix8_fft_kernel<<<blocks, threads, shared_mem, stream>>>(
        input, output, twiddle, n, direction
    );
    
    QFFT_CHECK(cudaGetLastError());
}

} // namespace QFFT
