// ============================================================================
// BIT REVERSAL PERMUTATION
// Optimized GPU implementation with warp shuffles
// ============================================================================

#include "../include/qfft.cuh"

namespace QFFT {

// ============================================================================
// BIT REVERSAL DEVICE FUNCTION
// ============================================================================

__device__ __forceinline__ 
unsigned int bit_reverse(unsigned int x, int bits) {
    unsigned int result = 0;
    #pragma unroll
    for (int i = 0; i < 20; ++i) {  // Max 2^20 = 1M elements
        if (i >= bits) break;
        if (x & (1 << i)) {
            result |= 1 << (bits - 1 - i);
        }
    }
    return result;
}

// ============================================================================
// BIT REVERSAL KERNEL WITH SHARED MEMORY
// ============================================================================

__global__ void bit_reverse_kernel(Complex32* data, size_t n, int log_n) {
    extern __shared__ Complex32 shared[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    if (idx < n) {
        // Compute bit-reversed index
        unsigned int rev_idx = bit_reverse(idx, log_n);
        
        // Load to shared memory
        shared[tid] = data[idx];
        __syncthreads();
        
        // Shuffle within block using warp primitives
        if (tid < WARP_SIZE) {
            int lane = tid % WARP_SIZE;
            int warp_id = tid / WARP_SIZE;
            
            // Use warp shuffle for fast permutation
            Complex32 val = shared[tid];
            unsigned int mask = 0xffffffff;
            
            // Butterfly shuffle pattern
            #pragma unroll
            for (int i = 0; i < 5; ++i) {  // log2(32) = 5
                int peer = lane ^ (1 << i);
                float temp_x = __shfl_sync(mask, val.x, peer);
                float temp_y = __shfl_sync(mask, val.y, peer);
                
                if ((lane & (1 << i)) != (peer & (1 << i))) {
                    val.x = temp_x;
                    val.y = temp_y;
                }
            }
            shared[tid] = val;
        }
        __syncthreads();
        
        // Write to global memory at reversed position
        if (idx < n && rev_idx < n && idx < rev_idx) {
            Complex32 temp = data[idx];
            data[idx] = data[rev_idx];
            data[rev_idx] = temp;
        }
    }
}

// ============================================================================
// LAUNCHER
// ============================================================================

void bit_reverse_gpu(Complex32* data, size_t n, cudaStream_t stream) {
    int log_n = log2_int(n);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    size_t shared_mem = threads * sizeof(Complex32);
    
    bit_reverse_kernel<<<blocks, threads, shared_mem, stream>>>(
        data, n, log_n
    );
    
    QFFT_CHECK(cudaGetLastError());
}

} // namespace QFFT
