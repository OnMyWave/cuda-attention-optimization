/*
 * Phase 5: LayerNorm CUDA Implementation
 * Uses warp-level reductions for efficient mean and variance computation
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define WARP_SIZE 32
#define EPS 1e-5f


/*
 * Warp-level reduction primitives
 */
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


/*
 * LayerNorm Kernel
 * Each block handles one sequence position
 * Threads within a block cooperate to compute mean and variance
 *
 * Input:
 *   x: [batch, seq_len, hidden_dim]
 *   gamma: [hidden_dim] - scale parameter
 *   beta: [hidden_dim] - shift parameter
 * Output:
 *   out: [batch, seq_len, hidden_dim]
 */
__global__ void layer_norm_kernel(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    int batch,
    int seq_len,
    int hidden_dim,
    float eps
) {
    // Each block processes one position
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;

    if (batch_idx >= batch || seq_idx >= seq_len) return;

    int offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;

    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        sum += x[offset + i];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Share across warps
    __shared__ float shared_sum[32];  // Max 32 warps
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction
    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total_sum += shared_sum[i];
        }
        shared_sum[0] = total_sum / hidden_dim;  // mean
    }
    __syncthreads();

    float mean = shared_sum[0];

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float diff = x[offset + i] - mean;
        var_sum += diff * diff;
    }

    // Warp-level reduction
    var_sum = warp_reduce_sum(var_sum);

    if (lane == 0) {
        shared_sum[warp_id] = var_sum;
    }
    __syncthreads();

    // Final reduction
    if (threadIdx.x == 0) {
        float total_var = 0.0f;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total_var += shared_sum[i];
        }
        shared_sum[0] = total_var / hidden_dim;  // variance
    }
    __syncthreads();

    float variance = shared_sum[0];
    float inv_std = rsqrtf(variance + eps);

    // Step 3: Normalize, scale, and shift
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float normalized = (x[offset + i] - mean) * inv_std;
        out[offset + i] = gamma[i] * normalized + beta[i];
    }
}


/*
 * Optimized LayerNorm for small hidden dimensions (e.g., 512, 768)
 * Uses shared memory more efficiently
 */
template<int HIDDEN_DIM>
__global__ void layer_norm_kernel_small(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    int batch,
    int seq_len,
    float eps
) {
    __shared__ float s_x[HIDDEN_DIM];
    __shared__ float s_gamma[HIDDEN_DIM];
    __shared__ float s_beta[HIDDEN_DIM];

    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;

    if (batch_idx >= batch || seq_idx >= seq_len) return;

    int offset = batch_idx * seq_len * HIDDEN_DIM + seq_idx * HIDDEN_DIM;

    // Load data into shared memory
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += blockDim.x) {
        s_x[i] = x[offset + i];
        s_gamma[i] = gamma[i];
        s_beta[i] = beta[i];
    }
    __syncthreads();

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += blockDim.x) {
        sum += s_x[i];
    }

    sum = warp_reduce_sum(sum);

    __shared__ float s_mean;
    __shared__ float s_inv_std;

    if (threadIdx.x == 0) {
        s_mean = sum / HIDDEN_DIM;
    }
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += blockDim.x) {
        float diff = s_x[i] - s_mean;
        var_sum += diff * diff;
    }

    var_sum = warp_reduce_sum(var_sum);

    if (threadIdx.x == 0) {
        float variance = var_sum / HIDDEN_DIM;
        s_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    // Normalize and write output
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += blockDim.x) {
        float normalized = (s_x[i] - s_mean) * s_inv_std;
        out[offset + i] = s_gamma[i] * normalized + s_beta[i];
    }
}


/*
 * Host function
 */
extern "C" {

void layer_norm_forward_cuda(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    int batch,
    int seq_len,
    int hidden_dim,
    float eps,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(seq_len, batch);

    // Use optimized kernel for common hidden dimensions
    if (hidden_dim == 512) {
        layer_norm_kernel_small<512><<<grid, block, 0, stream>>>(
            x, gamma, beta, out, batch, seq_len, eps
        );
    }
    else if (hidden_dim == 768) {
        layer_norm_kernel_small<768><<<grid, block, 0, stream>>>(
            x, gamma, beta, out, batch, seq_len, eps
        );
    }
    else if (hidden_dim == 1024) {
        layer_norm_kernel_small<1024><<<grid, block, 0, stream>>>(
            x, gamma, beta, out, batch, seq_len, eps
        );
    }
    else {
        // Generic kernel
        layer_norm_kernel<<<grid, block, 0, stream>>>(
            x, gamma, beta, out, batch, seq_len, hidden_dim, eps
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
