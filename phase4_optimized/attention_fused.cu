/*
 * Phase 4: Fused Attention Kernel
 * Combines Q@K^T, Softmax, and Attention@V into a single kernel
 * Minimizes global memory traffic by keeping intermediate results in shared memory/registers
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


/*
 * Fused Attention Kernel
 * Each block processes one query position
 *
 * Strategy:
 * 1. Load Q[i] for current query position into shared memory
 * 2. Iterate over K tiles:
 *    - Compute attention scores (Q[i] @ K^T)
 *    - Keep scores in shared memory
 * 3. Compute softmax online (without storing all scores)
 * 4. Iterate over V tiles:
 *    - Accumulate weighted sum directly to output
 *
 * This minimizes global memory writes by never materializing full score matrix
 */
template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void fused_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim,
    float scale
) {
    // Block processes one query position
    int batch_idx = blockIdx.y;
    int query_idx = blockIdx.x;

    if (batch_idx >= batch || query_idx >= seq_len) return;

    // Shared memory for current query
    __shared__ float s_query[HEAD_DIM];
    __shared__ float s_key[BLOCK_SIZE][HEAD_DIM];
    __shared__ float s_value[BLOCK_SIZE][HEAD_DIM];
    __shared__ float s_scores[BLOCK_SIZE];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Load query into shared memory
    int q_offset = batch_idx * seq_len * head_dim + query_idx * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        s_query[d] = Q[q_offset + d];
    }
    __syncthreads();

    // Initialize output accumulator
    float output_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        output_acc[d] = 0.0f;
    }

    // Online softmax statistics
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Process keys and values in blocks
    int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int key_start = block_idx * BLOCK_SIZE;
        int block_keys = min(BLOCK_SIZE, seq_len - key_start);

        // Load keys for this block
        for (int k = tid; k < block_keys; k += num_threads) {
            int key_idx = key_start + k;
            int k_offset = batch_idx * seq_len * head_dim + key_idx * head_dim;

            for (int d = 0; d < head_dim; d++) {
                s_key[k][d] = K[k_offset + d];
            }
        }
        __syncthreads();

        // Compute scores for this block: Q @ K^T
        for (int k = tid; k < block_keys; k += num_threads) {
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < head_dim; d++) {
                score += s_query[d] * s_key[k][d];
            }
            s_scores[k] = score * scale;
        }
        __syncthreads();

        // Update online softmax statistics
        float block_max = -INFINITY;
        for (int k = tid; k < block_keys; k += num_threads) {
            block_max = fmaxf(block_max, s_scores[k]);
        }

        // Warp-level reduction for max
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
        }

        // Share max across warps
        __shared__ float s_max[32];
        int lane = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        if (lane == 0) {
            s_max[warp_id] = block_max;
        }
        __syncthreads();

        if (tid == 0) {
            block_max = s_max[0];
            for (int i = 1; i < (num_threads + WARP_SIZE - 1) / WARP_SIZE; i++) {
                block_max = fmaxf(block_max, s_max[i]);
            }
            s_max[0] = block_max;
        }
        __syncthreads();
        block_max = s_max[0];

        // Update global max and rescale previous sum
        float old_max = max_score;
        max_score = fmaxf(max_score, block_max);
        float rescale = expf(old_max - max_score);
        sum_exp *= rescale;

        // Rescale output accumulator
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            output_acc[d] *= rescale;
        }

        // Load values for this block
        for (int k = tid; k < block_keys; k += num_threads) {
            int key_idx = key_start + k;
            int v_offset = batch_idx * seq_len * head_dim + key_idx * head_dim;

            for (int d = 0; d < head_dim; d++) {
                s_value[k][d] = V[v_offset + d];
            }
        }
        __syncthreads();

        // Compute attention weights and accumulate output
        for (int k = tid; k < block_keys; k += num_threads) {
            float attn_weight = expf(s_scores[k] - max_score);
            sum_exp += attn_weight;

            #pragma unroll
            for (int d = 0; d < head_dim; d++) {
                output_acc[d] += attn_weight * s_value[k][d];
            }
        }
        __syncthreads();
    }

    // Reduce sum_exp across threads
    __shared__ float s_sum[32];
    float thread_sum = sum_exp;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if (lane == 0) {
        s_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total_sum = s_sum[0];
        for (int i = 1; i < (num_threads + WARP_SIZE - 1) / WARP_SIZE; i++) {
            total_sum += s_sum[i];
        }
        s_sum[0] = total_sum;
    }
    __syncthreads();
    float total_sum = s_sum[0];

    // Normalize and write output
    int out_offset = batch_idx * seq_len * head_dim + query_idx * head_dim;
    for (int d = tid; d < head_dim; d += num_threads) {
        atomicAdd(&out[out_offset + d], output_acc[d] / total_sum);
    }
}


/*
 * Simplified fused kernel for small head dimensions
 * Each thread handles one query position
 */
__global__ void fused_attention_simple_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim,
    float scale
) {
    int batch_idx = blockIdx.y;
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch || query_idx >= seq_len) return;

    int q_offset = batch_idx * seq_len * head_dim + query_idx * head_dim;

    // Step 1: Compute all attention scores and find max
    float max_score = -INFINITY;

    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        int k_offset = batch_idx * seq_len * head_dim + k_idx * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score *= scale;

        max_score = fmaxf(max_score, score);
    }

    // Step 2: Compute exp and sum (second pass needed for numerical stability)
    float sum_exp = 0.0f;

    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        int k_offset = batch_idx * seq_len * head_dim + k_idx * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score *= scale;

        sum_exp += expf(score - max_score);
    }

    // Step 3: Compute weighted sum with V
    int out_offset = batch_idx * seq_len * head_dim + query_idx * head_dim;

    for (int d = 0; d < head_dim; d++) {
        float output_val = 0.0f;

        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            int k_offset = batch_idx * seq_len * head_dim + k_idx * head_dim;
            int v_offset = batch_idx * seq_len * head_dim + k_idx * head_dim;

            float score = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                score += Q[q_offset + dd] * K[k_offset + dd];
            }
            score *= scale;

            float attn_weight = expf(score - max_score) / sum_exp;
            output_val += attn_weight * V[v_offset + d];
        }

        out[out_offset + d] = output_val;
    }
}


/*
 * Host function
 */
extern "C" {

void attention_forward_fused_cuda(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Initialize output to zero
    size_t out_size = batch * seq_len * head_dim * sizeof(float);
    CUDA_CHECK(cudaMemsetAsync(out, 0, out_size, stream));

    if (head_dim == 64 && seq_len <= 512) {
        // Use optimized fused kernel
        const int BLOCK_SIZE = 128;
        const int HEAD_DIM = 64;

        dim3 block(256);
        dim3 grid(seq_len, batch);

        fused_attention_kernel<BLOCK_SIZE, HEAD_DIM><<<grid, block, 0, stream>>>(
            Q, K, V, out, batch, seq_len, head_dim, scale
        );
        CUDA_CHECK(cudaGetLastError());
    }
    else {
        // Use simple kernel (slower but works for any size)
        dim3 block(256);
        dim3 grid((seq_len + block.x - 1) / block.x, batch);

        fused_attention_simple_kernel<<<grid, block, 0, stream>>>(
            Q, K, V, out, batch, seq_len, head_dim, scale
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

} // extern "C"
