/*
 * Phase 2: Naive CUDA Implementation of Attention Kernels
 * Three separate kernels: Q@K^T, Softmax, Attention@V
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
 * Kernel 1: Matrix Multiplication Q @ K^T
 * Input:
 *   Q: [batch, seq_len, head_dim]
 *   K: [batch, seq_len, head_dim]
 * Output:
 *   scores: [batch, seq_len, seq_len]
 */
__global__ void matmul_qk_kernel(
    const float* Q,
    const float* K,
    float* scores,
    int batch,
    int seq_len,
    int head_dim,
    float scale
) {
    // Each thread computes one element of scores
    int b = blockIdx.z;                           // batch index
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // query position
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // key position

    if (b >= batch || row >= seq_len || col >= seq_len) {
        return;
    }

    // Compute dot product: Q[b, row, :] · K[b, col, :]
    float sum = 0.0f;
    int q_offset = b * seq_len * head_dim + row * head_dim;
    int k_offset = b * seq_len * head_dim + col * head_dim;

    for (int d = 0; d < head_dim; d++) {
        sum += Q[q_offset + d] * K[k_offset + d];
    }

    // Apply scaling and write output
    int out_idx = b * seq_len * seq_len + row * seq_len + col;
    scores[out_idx] = sum * scale;
}


/*
 * Kernel 2: Softmax (per-row)
 * Input:
 *   scores: [batch, seq_len, seq_len]
 * Output:
 *   attn: [batch, seq_len, seq_len]
 */
__global__ void softmax_kernel(
    const float* scores,
    float* attn,
    int batch,
    int seq_len
) {
    // Each thread handles one row (one query position)
    int b = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch || row >= seq_len) {
        return;
    }

    int offset = b * seq_len * seq_len + row * seq_len;

    // Step 1: Find max for numerical stability
    float max_val = -INFINITY;
    for (int col = 0; col < seq_len; col++) {
        float val = scores[offset + col];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int col = 0; col < seq_len; col++) {
        float val = expf(scores[offset + col] - max_val);
        attn[offset + col] = val;
        sum += val;
    }

    // Step 3: Normalize
    for (int col = 0; col < seq_len; col++) {
        attn[offset + col] /= sum;
    }
}


/*
 * Kernel 3: Matrix Multiplication Attention @ V
 * Input:
 *   attn: [batch, seq_len, seq_len]
 *   V: [batch, seq_len, head_dim]
 * Output:
 *   out: [batch, seq_len, head_dim]
 */
__global__ void matmul_av_kernel(
    const float* attn,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim
) {
    // Each thread computes one element of output
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // query position
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // feature dimension

    if (b >= batch || row >= seq_len || col >= head_dim) {
        return;
    }

    // Compute weighted sum: attn[b, row, :] · V[b, :, col]
    float sum = 0.0f;
    int attn_offset = b * seq_len * seq_len + row * seq_len;

    for (int k = 0; k < seq_len; k++) {
        int v_idx = b * seq_len * head_dim + k * head_dim + col;
        sum += attn[attn_offset + k] * V[v_idx];
    }

    // Write output
    int out_idx = b * seq_len * head_dim + row * head_dim + col;
    out[out_idx] = sum;
}

/*
 * Host function: Launch all three kernels
 */
extern "C" {

void attention_forward_cuda(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    // Allocate intermediate buffers
    float* scores;
    float* attn;

    size_t scores_size = batch * seq_len * seq_len * sizeof(float);
    CUDA_CHECK(cudaMalloc(&scores, scores_size));
    CUDA_CHECK(cudaMalloc(&attn, scores_size));

    // Kernel 1: Q @ K^T
    float scale = 1.0f / sqrtf((float)head_dim);

    dim3 block1(16, 16);
    dim3 grid1(
        (seq_len + block1.x - 1) / block1.x,
        (seq_len + block1.y - 1) / block1.y,
        batch
    );

    matmul_qk_kernel<<<grid1, block1, 0, stream>>>(
        Q, K, scores, batch, seq_len, head_dim, scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 추가
    printf("Kernel 1 (Q@K^T) completed\n");  // 추가

    // Kernel 2: Softmax
    dim3 block2(256);
    dim3 grid2((seq_len + block2.x - 1) / block2.x, batch);

    softmax_kernel<<<grid2, block2, 0, stream>>>(
        scores, attn, batch, seq_len
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 추가
    printf("Kernel 2 (Softmax) completed\n");  // 추가

    // Kernel 3: Attention @ V
    dim3 block3(16, 16);
    dim3 grid3(
        (head_dim + block3.x - 1) / block3.x,
        (seq_len + block3.y - 1) / block3.y,
        batch
    );

    printf("Launching Kernel 3: grid(%d, %d, %d), block(%d, %d)\n",
           grid3.x, grid3.y, grid3.z, block3.x, block3.y);  // 추가

    matmul_av_kernel<<<grid3, block3, 0, stream>>>(
        attn, V, out, batch, seq_len, head_dim
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 추가
    printf("Kernel 3 (Attn@V) completed\n");  // 추가

    // Free intermediate buffers
    CUDA_CHECK(cudaFree(scores));
    CUDA_CHECK(cudaFree(attn));
}

} // extern "C"