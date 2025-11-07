/*
 * Phase 5: MLP (Feed-Forward Network) CUDA Implementation
 * Uses cuBLAS for matrix multiplications and custom GELU kernel
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
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
 * GELU device function
 */
__device__ inline float gelu(float x) {
    const float c = 0.797884560804236f;  // sqrt(2/pi)
    const float a = 0.044715f;
    float x_cubed = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + a * x_cubed)));
}

/*
 * GELU Activation Kernel
 * Applies GELU element-wise: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
__global__ void gelu_kernel(
    const float* input,
    float* output,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        output[idx] = gelu(input[idx]);
    }
}


/*
 * Add bias kernel
 */
__global__ void add_bias_kernel(
    float* data,
    const float* bias,
    int batch_seq,
    int dim
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    if (col < dim && row < batch_seq) {
        data[row * dim + col] += bias[col];
    }
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
 * Host function using cuBLAS
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
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    int batch_seq = batch * seq_len;

    // Allocate intermediate buffer
    float* intermediate;
    size_t inter_size = batch_seq * ff_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&intermediate, inter_size));

    // Step 1: Linear1 - x @ W1^T
    // x: [batch_seq, hidden_dim]
    // W1: [hidden_dim, ff_dim] stored as row-major
    // Result: [batch_seq, ff_dim]
    // We want: intermediate = x @ W1^T

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major layout
    // Our tensors are row-major, so we compute: C^T = B^T @ A^T
    // where C = A @ B in row-major becomes C^T = B^T @ A^T in column-major
    // Here: intermediate = x @ W1^T, so intermediate^T = W1 @ x^T
    cublasSgemm(handle,
        CUBLAS_OP_N,        // W1 not transposed (but will be seen as W1^T due to row-major)
        CUBLAS_OP_N,        // x not transposed (but will be seen as x^T due to row-major)
        ff_dim,             // rows of result in column-major = cols in row-major
        batch_seq,          // cols of result in column-major = rows in row-major
        hidden_dim,         // common dimension
        &alpha,
        W1, ff_dim,         // leading dimension in column-major view
        x, hidden_dim,      // leading dimension in column-major view
        &beta,
        intermediate, ff_dim
    );

    // Add bias1
    dim3 block_bias1(256);
    dim3 grid_bias1((ff_dim + block_bias1.x - 1) / block_bias1.x, batch_seq);
    add_bias_kernel<<<grid_bias1, block_bias1, 0, stream>>>(
        intermediate, b1, batch_seq, ff_dim
    );

    // Step 2: Apply GELU activation
    int total_elements = batch_seq * ff_dim;
    dim3 block_gelu(256);
    dim3 grid_gelu((total_elements + block_gelu.x - 1) / block_gelu.x);
    gelu_kernel<<<grid_gelu, block_gelu, 0, stream>>>(
        intermediate, intermediate, total_elements
    );

    // Step 3: Linear2 - intermediate @ W2^T
    // intermediate: [batch_seq, ff_dim]
    // W2: [ff_dim, hidden_dim] stored as row-major
    // Result: [batch_seq, hidden_dim]
    // We want: out = intermediate @ W2^T

    // Similar to step 1: out^T = W2 @ intermediate^T
    cublasSgemm(handle,
        CUBLAS_OP_N,        // W2 not transposed
        CUBLAS_OP_N,        // intermediate not transposed
        hidden_dim,         // rows of result in column-major = cols in row-major
        batch_seq,          // cols of result in column-major = rows in row-major
        ff_dim,             // common dimension
        &alpha,
        W2, hidden_dim,     // leading dimension in column-major view
        intermediate, ff_dim, // leading dimension in column-major view
        &beta,
        out, hidden_dim
    );

    // Add bias2
    dim3 block_bias2(256);
    dim3 grid_bias2((hidden_dim + block_bias2.x - 1) / block_bias2.x, batch_seq);
    add_bias_kernel<<<grid_bias2, block_bias2, 0, stream>>>(
        out, b2, batch_seq, hidden_dim
    );

    // Cleanup
    CUDA_CHECK(cudaFree(intermediate));
    cublasDestroy(handle);
}

} // extern "C"
