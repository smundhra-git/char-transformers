#pragma once

#include "tensor.hpp"

namespace engine {

    // Element-wise ops
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor sub(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b); // element-wise
    Tensor scale(const Tensor& a, double alpha);
    
    // Matrix/Batch ops
    // Matmul supports broadcasting:
    // (..., M, K) @ (..., K, N) -> (..., M, N)
    Tensor matmul(const Tensor& a, const Tensor& b);

    // Transpose last two dimensions
    Tensor transpose(const Tensor& a);
    
    // Permute dimensions (generic transpose)
    Tensor permute(const Tensor& a, const std::vector<size_t>& dims);

    // Reshape (differentiable view)
    Tensor reshape(const Tensor& a, const Shape& shape);

    // Activations
    Tensor relu(const Tensor& x);
    Tensor softmax(const Tensor& x, int dim = -1);
    Tensor gelu(const Tensor& x); // Gaussian Error Linear Unit (often used in Transformers)

    // Normalization
    Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps = 1e-5);

    // Loss
    Tensor cross_entropy(const Tensor& logits, const Tensor& target); // logits: (B, T, V), target: (B, T) indices
}
