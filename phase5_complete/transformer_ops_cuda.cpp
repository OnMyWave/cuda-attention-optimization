/*
 * PyTorch C++ Extension for LayerNorm and MLP
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declarations
extern "C" void layer_norm_forward_cuda(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    int batch,
    int seq_len,
    int hidden_dim,
    float eps,
    cudaStream_t stream
);

extern "C" void mlp_forward_cuda(
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
);


torch::Tensor layer_norm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps = 1e-5
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");

    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, seq_len, hidden_dim]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D [hidden_dim]");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D [hidden_dim]");

    int batch = x.size(0);
    int seq_len = x.size(1);
    int hidden_dim = x.size(2);

    TORCH_CHECK(gamma.size(0) == hidden_dim, "gamma size must match hidden_dim");
    TORCH_CHECK(beta.size(0) == hidden_dim, "beta size must match hidden_dim");

    auto out = torch::empty_like(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    layer_norm_forward_cuda(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        hidden_dim,
        eps,
        stream
    );

    return out;
}


torch::Tensor mlp_forward(
    torch::Tensor x,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(W1.is_cuda(), "W1 must be a CUDA tensor");
    TORCH_CHECK(b1.is_cuda(), "b1 must be a CUDA tensor");
    TORCH_CHECK(W2.is_cuda(), "W2 must be a CUDA tensor");
    TORCH_CHECK(b2.is_cuda(), "b2 must be a CUDA tensor");

    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, seq_len, hidden_dim]");
    TORCH_CHECK(W1.dim() == 2, "W1 must be 2D [hidden_dim, ff_dim]");
    TORCH_CHECK(b1.dim() == 1, "b1 must be 1D [ff_dim]");
    TORCH_CHECK(W2.dim() == 2, "W2 must be 2D [ff_dim, hidden_dim]");
    TORCH_CHECK(b2.dim() == 1, "b2 must be 1D [hidden_dim]");

    int batch = x.size(0);
    int seq_len = x.size(1);
    int hidden_dim = x.size(2);
    int ff_dim = W1.size(1);

    TORCH_CHECK(W1.size(0) == hidden_dim, "W1 input dimension must match hidden_dim");
    TORCH_CHECK(b1.size(0) == ff_dim, "b1 size must match ff_dim");
    TORCH_CHECK(W2.size(0) == ff_dim, "W2 input dimension must match ff_dim");
    TORCH_CHECK(W2.size(1) == hidden_dim, "W2 output dimension must match hidden_dim");
    TORCH_CHECK(b2.size(0) == hidden_dim, "b2 size must match hidden_dim");

    auto out = torch::empty_like(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mlp_forward_cuda(
        x.data_ptr<float>(),
        W1.data_ptr<float>(),
        b1.data_ptr<float>(),
        W2.data_ptr<float>(),
        b2.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        seq_len,
        hidden_dim,
        ff_dim,
        stream
    );

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm", &layer_norm_forward, "LayerNorm forward (CUDA)",
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);

    m.def("mlp_forward", &mlp_forward, "MLP forward (CUDA)",
          py::arg("x"), py::arg("W1"), py::arg("b1"), py::arg("W2"), py::arg("b2"));
}
