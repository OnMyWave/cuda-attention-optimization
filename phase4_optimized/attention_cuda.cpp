/*
 * PyTorch C++ Extension for Phase 4 Optimized Attention
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Forward declaration of CUDA function
extern "C" void attention_forward_fused_cuda(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim,
    int tile_size,
    cudaStream_t stream
);


torch::Tensor attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int tile_size = 16
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
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Launch CUDA kernels
    attention_forward_fused_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        head_dim,
        tile_size,
        stream
    );

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Fused Attention forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("tile_size") = 16);
}
