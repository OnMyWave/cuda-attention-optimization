/*
 * Phase 3: Tiled CUDA Implementation with Shared Memory
 * Optimized matrix multiplication using shared memory tiling
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

// Tile size (tunable parameter)
#define TILE_SIZE 16


/*
 * Kernel 1: Tiled Matrix Multiplication Q @ K^T
 * Uses shared memory for better memory bandwidth utilization
 */
template<int TILE>
__global__ void matmul_qk_tiled_kernel(
    const float* Q,
    const float* K,
    float* scores,
    int batch,
    int seq_len,
    int head_dim,
    float scale
) {
    // Shared memory for tiles
    __shared__ float Q_tile[TILE][TILE];
    __shared__ float K_tile[TILE][TILE];

    int b = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if (b >= batch) return;

    float sum = 0.0f;

    // Loop over tiles of the K dimension (head_dim)
    int num_tiles = (head_dim + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        // Load Q tile into shared memory
        int q_row = row;
        int q_col = t * TILE + threadIdx.x;

        if (q_row < seq_len && q_col < head_dim) {
            int q_idx = b * seq_len * head_dim + q_row * head_dim + q_col;
            Q_tile[threadIdx.y][threadIdx.x] = Q[q_idx];
        } else {
            Q_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load K tile into shared memory (K^T)
        int k_row = col;  // Note: transposed
        int k_col = t * TILE + threadIdx.y;

        if (k_row < seq_len && k_col < head_dim) {
            int k_idx = b * seq_len * head_dim + k_row * head_dim + k_col;
            K_tile[threadIdx.y][threadIdx.x] = K[k_idx];
        } else {
            K_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += Q_tile[threadIdx.y][k] * K_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result with scaling
    if (row < seq_len && col < seq_len) {
        int out_idx = b * seq_len * seq_len + row * seq_len + col;
        scores[out_idx] = sum * scale;
    }
}


/*
 * Kernel 2: Optimized Softmax with warp-level reductions
 */
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_optimized_kernel(
    const float* scores,
    float* attn,
    int batch,
    int seq_len
) {
    // Each block handles one row
    int b = blockIdx.y;
    int row = blockIdx.x;

    if (b >= batch || row >= seq_len) return;

    int offset = b * seq_len * seq_len + row * seq_len;

    // Step 1: Find max using warp reduction
    float max_val = -INFINITY;

    for (int col = threadIdx.x; col < seq_len; col += blockDim.x) {
        float val = scores[offset + col];
        max_val = fmaxf(max_val, val);
    }

    // Warp-level reduction for max
    max_val = warp_reduce_max(max_val);

    // Share max across warps
    __shared__ float shared_max[32];  // Max 32 warps per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();

    // Final reduction across warps
    if (threadIdx.x == 0) {
        max_val = shared_max[0];
        for (int i = 1; i < (blockDim.x + 31) / 32; i++) {
            max_val = fmaxf(max_val, shared_max[i]);
        }
        shared_max[0] = max_val;
    }
    __syncthreads();

    max_val = shared_max[0];

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;

    for (int col = threadIdx.x; col < seq_len; col += blockDim.x) {
        float val = expf(scores[offset + col] - max_val);
        attn[offset + col] = val;
        sum += val;
    }

    // Warp-level reduction for sum
    sum = warp_reduce_sum(sum);

    __shared__ float shared_sum[32];
    if (lane == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction for sum
    if (threadIdx.x == 0) {
        sum = shared_sum[0];
        for (int i = 1; i < (blockDim.x + 31) / 32; i++) {
            sum += shared_sum[i];
        }
        shared_sum[0] = sum;
    }
    __syncthreads();

    sum = shared_sum[0];

    // Step 3: Normalize
    for (int col = threadIdx.x; col < seq_len; col += blockDim.x) {
        attn[offset + col] /= sum;
    }
}


/*
 * Kernel 3: Tiled Matrix Multiplication Attention @ V
 */
template<int TILE>
__global__ void matmul_av_tiled_kernel(
    const float* attn,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim
) {
    __shared__ float attn_tile[TILE][TILE];
    __shared__ float V_tile[TILE][TILE];

    int b = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if (b >= batch) return;

    float sum = 0.0f;

    int num_tiles = (seq_len + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        // Load attention tile
        int a_row = row;
        int a_col = t * TILE + threadIdx.x;

        if (a_row < seq_len && a_col < seq_len) {
            int a_idx = b * seq_len * seq_len + a_row * seq_len + a_col;
            attn_tile[threadIdx.y][threadIdx.x] = attn[a_idx];
        } else {
            attn_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load V tile
        int v_row = t * TILE + threadIdx.y;
        int v_col = col;

        if (v_row < seq_len && v_col < head_dim) {
            int v_idx = b * seq_len * head_dim + v_row * head_dim + v_col;
            V_tile[threadIdx.y][threadIdx.x] = V[v_idx];
        } else {
            V_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += attn_tile[threadIdx.y][k] * V_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < seq_len && col < head_dim) {
        int out_idx = b * seq_len * head_dim + row * head_dim + col;
        out[out_idx] = sum;
    }
}


/*
 * Host function: Launch tiled kernels
 */
extern "C" {

void attention_forward_tiled_cuda(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim,
    int tile_size,
    cudaStream_t stream
) {
    // Allocate intermediate buffers
    float* scores;
    float* attn;

    size_t scores_size = batch * seq_len * seq_len * sizeof(float);
    CUDA_CHECK(cudaMalloc(&scores, scores_size));
    CUDA_CHECK(cudaMalloc(&attn, scores_size));

    float scale = 1.0f / sqrtf((float)head_dim);

    // Select kernel based on tile size
    if (tile_size == 16) {
        // Kernel 1: Q @ K^T with tiling
        dim3 block1(TILE_SIZE, TILE_SIZE);
        dim3 grid1(
            (seq_len + TILE_SIZE - 1) / TILE_SIZE,
            (seq_len + TILE_SIZE - 1) / TILE_SIZE,
            batch
        );

        matmul_qk_tiled_kernel<TILE_SIZE><<<grid1, block1, 0, stream>>>(
            Q, K, scores, batch, seq_len, head_dim, scale
        );
        CUDA_CHECK(cudaGetLastError());

        // Kernel 2: Softmax
        dim3 block2(256);
        dim3 grid2(seq_len, batch);

        softmax_optimized_kernel<<<grid2, block2, 0, stream>>>(
            scores, attn, batch, seq_len
        );
        CUDA_CHECK(cudaGetLastError());

        // Kernel 3: Attention @ V with tiling
        dim3 block3(TILE_SIZE, TILE_SIZE);
        dim3 grid3(
            (head_dim + TILE_SIZE - 1) / TILE_SIZE,
            (seq_len + TILE_SIZE - 1) / TILE_SIZE,
            batch
        );

        matmul_av_tiled_kernel<TILE_SIZE><<<grid3, block3, 0, stream>>>(
            attn, V, out, batch, seq_len, head_dim
        );
        CUDA_CHECK(cudaGetLastError());
    }
    // Add support for other tile sizes (8, 32) if needed
    else {
        fprintf(stderr, "Unsupported tile size: %d. Using default 16.\n", tile_size);

        dim3 block1(TILE_SIZE, TILE_SIZE);
        dim3 grid1(
            (seq_len + TILE_SIZE - 1) / TILE_SIZE,
            (seq_len + TILE_SIZE - 1) / TILE_SIZE,
            batch
        );

        matmul_qk_tiled_kernel<TILE_SIZE><<<grid1, block1, 0, stream>>>(
            Q, K, scores, batch, seq_len, head_dim, scale
        );

        dim3 block2(256);
        dim3 grid2(seq_len, batch);
        softmax_optimized_kernel<<<grid2, block2, 0, stream>>>(
            scores, attn, batch, seq_len
        );

        dim3 block3(TILE_SIZE, TILE_SIZE);
        dim3 grid3(
            (head_dim + TILE_SIZE - 1) / TILE_SIZE,
            (seq_len + TILE_SIZE - 1) / TILE_SIZE,
            batch
        );

        matmul_av_tiled_kernel<TILE_SIZE><<<grid3, block3, 0, stream>>>(
            attn, V, out, batch, seq_len, head_dim
        );
    }

    // Free intermediate buffers
    CUDA_CHECK(cudaFree(scores));
    CUDA_CHECK(cudaFree(attn));
}

} // extern "C"
