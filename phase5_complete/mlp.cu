/*
 * Phase 5: MLP (Feed-Forward Network) CUDA Implementation
 * Fuses Linear1 → GELU → Linear2 into fewer kernel launches
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


/*
 * GELU Activation Function
 * Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
__device__ inline float gelu(float x) {
    const float c = 0.797884560804236f;  // sqrt(2/pi)
    const float a = 0.044715f;
    float x_cubed = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + a * x_cubed)));
}


/*
 * Fused MLP Kernel: Linear1 → GELU → Linear2
 *
 * Each block processes one sequence position
 * Uses shared memory for weight tiles
 *
 * Input:
 *   x: [batch, seq_len, hidden_dim]
 *   W1: [hidden_dim, ff_dim]
 *   b1: [ff_dim]
 *   W2: [ff_dim, hidden_dim]
 *   b2: [hidden_dim]
 * Output:
 *   out: [batch, seq_len, hidden_dim]
 */
template<int HIDDEN_DIM, int FF_DIM, int TILE_SIZE>
__global__ void fused_mlp_kernel(
    const float* x,
    const float* W1,
    const float* b1,
    const float* W2,
    const float* b2,
    float* out,
    int batch,
    int seq_len
) {
    __shared__ float s_x[HIDDEN_DIM];
    __shared__ float s_intermediate[FF_DIM];

    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;

    if (batch_idx >= batch || seq_idx >= seq_len) return;

    int x_offset = batch_idx * seq_len * HIDDEN_DIM + seq_idx * HIDDEN_DIM;
    int out_offset = batch_idx * seq_len * HIDDEN_DIM + seq_idx * HIDDEN_DIM;

    // Load input into shared memory
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += blockDim.x) {
        s_x[i] = x[x_offset + i];
    }
    __syncthreads();

    // Layer 1: x @ W1 + b1 → GELU
    for (int i = threadIdx.x; i < FF_DIM; i += blockDim.x) {
        float sum = b1[i];

        // Matrix multiplication: x @ W1[:, i]
        #pragma unroll
        for (int j = 0; j < HIDDEN_DIM; j++) {
            sum += s_x[j] * W1[j * FF_DIM + i];
        }

        // Apply GELU activation
        s_intermediate[i] = gelu(sum);
    }
    __syncthreads();

    // Layer 2: intermediate @ W2 + b2
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += blockDim.x) {
        float sum = b2[i];

        // Matrix multiplication: intermediate @ W2[:, i]
        #pragma unroll
        for (int j = 0; j < FF_DIM; j++) {
            sum += s_intermediate[j] * W2[j * HIDDEN_DIM + i];
        }

        out[out_offset + i] = sum;
    }
}


/*
 * Generic MLP kernel (for any dimension sizes)
 * Separate kernels for each stage
 */
__global__ void linear_gelu_kernel(
    const float* x,
    const float* W,
    const float* b,
    float* out,
    int batch,
    int seq_len,
    int in_dim,
    int out_dim
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch || seq_idx >= seq_len || out_idx >= out_dim) return;

    int x_offset = batch_idx * seq_len * in_dim + seq_idx * in_dim;
    int output_offset = batch_idx * seq_len * out_dim + seq_idx * out_dim;

    // Compute one output element
    float sum = b[out_idx];

    for (int i = 0; i < in_dim; i++) {
        sum += x[x_offset + i] * W[i * out_dim + out_idx];
    }

    // Apply GELU
    out[output_offset + out_idx] = gelu(sum);
}


__global__ void linear_kernel(
    const float* x,
    const float* W,
    const float* b,
    float* out,
    int batch,
    int seq_len,
    int in_dim,
    int out_dim
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch || seq_idx >= seq_len || out_idx >= out_dim) return;

    int x_offset = batch_idx * seq_len * in_dim + seq_idx * in_dim;
    int output_offset = batch_idx * seq_len * out_dim + seq_idx * out_dim;

    // Compute one output element
    float sum = b[out_idx];

    for (int i = 0; i < in_dim; i++) {
        sum += x[x_offset + i] * W[i * out_dim + out_idx];
    }

    out[output_offset + out_idx] = sum;
}


/*
 * Host function
 */
extern "C" {

void mlp_forward_cuda(
    const float* x,
    const float* W1,
    const float* b1,
    const float* W2,
    const float* b2,
    float* out,
    int batch,
    int seq_len,
    int hidden_dim,
    int ff_dim,
    cudaStream_t stream
) {
    // Use fused kernel for common dimensions
    if (hidden_dim == 512 && ff_dim == 2048) {
        dim3 block(256);
        dim3 grid(seq_len, batch);

        fused_mlp_kernel<512, 2048, 16><<<grid, block, 0, stream>>>(
            x, W1, b1, W2, b2, out, batch, seq_len
        );
        CUDA_CHECK(cudaGetLastError());
    }
    else {
        // Generic implementation: two separate kernels
        // Allocate intermediate buffer
        float* intermediate;
        size_t inter_size = batch * seq_len * ff_dim * sizeof(float);
        CUDA_CHECK(cudaMalloc(&intermediate, inter_size));

        // Kernel 1: Linear + GELU
        dim3 block1(256);
        dim3 grid1(
            (ff_dim + block1.x - 1) / block1.x,
            seq_len,
            batch
        );

        linear_gelu_kernel<<<grid1, block1, 0, stream>>>(
            x, W1, b1, intermediate, batch, seq_len, hidden_dim, ff_dim
        );
        CUDA_CHECK(cudaGetLastError());

        // Kernel 2: Linear (no activation)
        dim3 block2(256);
        dim3 grid2(
            (hidden_dim + block2.x - 1) / block2.x,
            seq_len,
            batch
        );

        linear_kernel<<<grid2, block2, 0, stream>>>(
            intermediate, W2, b2, out, batch, seq_len, ff_dim, hidden_dim
        );
        CUDA_CHECK(cudaGetLastError());

        // Free intermediate buffer
        CUDA_CHECK(cudaFree(intermediate));
    }
}

} // extern "C"
