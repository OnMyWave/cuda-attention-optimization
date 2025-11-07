/*
 * PyTorch C++ Extension for Naive CUDA Attention
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Forward declaration of CUDA function
extern "C" void attention_forward_cuda(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim,
    cudaStream_t stream
);


/*
 * PyTorch wrapper for attention forward pass
 * Input:
 *   Q: [batch, seq_len, head_dim]
 *   K: [batch, seq_len, head_dim]
 *   V: [batch, seq_len, head_dim]
 * Output:
 *   out: [batch, seq_len, head_dim]
 */
torch::Tensor attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // Check inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    TORCH_CHECK(Q.dim() == 3, "Q must be 3D");
    TORCH_CHECK(K.dim() == 3, "K must be 3D");
    TORCH_CHECK(V.dim() == 3, "V must be 3D");

    TORCH_CHECK(Q.size(0) == K.size(0) && K.size(0) == V.size(0), "Batch size must match");
    TORCH_CHECK(Q.size(1) == K.size(1) && K.size(1) == V.size(1), "Sequence length must match");
    TORCH_CHECK(Q.size(2) == K.size(2) && K.size(2) == V.size(2), "Head dimension must match");

    // Get dimensions
    int batch = Q.size(0);
    int seq_len = Q.size(1);
    int head_dim = Q.size(2);

    // Allocate output tensor
    auto out = torch::empty_like(Q);

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch CUDA kernels
    attention_forward_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        head_dim,
        stream
    );

    return out;
}


/*
 * Separate kernel wrappers for testing
 */

// Q @ K^T
torch::Tensor matmul_qk(
    torch::Tensor Q,
    torch::Tensor K,
    float scale
);

// Softmax
torch::Tensor softmax(
    torch::Tensor scores
);

// Attention @ V
torch::Tensor matmul_av(
    torch::Tensor attn,
    torch::Tensor V
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Attention forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"));

    // Optional: export individual kernels for testing
    // m.def("matmul_qk", &matmul_qk, "Q @ K^T");
    // m.def("softmax", &softmax, "Softmax");
    // m.def("matmul_av", &matmul_av, "Attention @ V");
}
