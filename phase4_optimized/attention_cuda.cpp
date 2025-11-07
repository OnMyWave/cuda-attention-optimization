/*
 * PyTorch C++ Extension for Fused CUDA Attention
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" void attention_forward_fused_cuda(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch,
    int seq_len,
    int head_dim,
    cudaStream_t stream
);


torch::Tensor attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    TORCH_CHECK(Q.dim() == 3, "Q must be 3D");
    TORCH_CHECK(K.dim() == 3, "K must be 3D");
    TORCH_CHECK(V.dim() == 3, "V must be 3D");

    int batch = Q.size(0);
    int seq_len = Q.size(1);
    int head_dim = Q.size(2);

    auto out = torch::empty_like(Q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    attention_forward_fused_cuda(
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Fused Attention forward (CUDA)");
}
