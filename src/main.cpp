#include <iostream>
#include <cstddef>

#include "math/matrix.hpp"
#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "nn/self_attention.hpp"
#include "nn/linear.hpp"

using namespace std;
using namespace math;
using namespace engine;
using namespace nn;

int main() {
    cout << "=== SelfAttention test ===" << endl;

    // sequence length T, model dimension d_model
    std::size_t T = 4;
    std::size_t d_model = 8;

    // 1) Build a simple input X: [T x d_model]
    Matrix X_m(T, d_model, 0.0);
    for (std::size_t i = 0; i < T; ++i) {
        for (std::size_t j = 0; j < d_model; ++j) {
            X_m.data[i * d_model + j] = static_cast<double>(i + j);
        }
    }

    // Wrap as Tensor with gradients
    Tensor x(X_m, /*require_grad=*/true);

    // 2) Create self-attention layer (single-head, causal)
    SelfAttentionConfig cfg{ d_model, /*causal=*/true };
    SelfAttention attn(cfg);

    // 3) Forward pass: y = SA(x), y: [T x d_model]
    Tensor y = attn.forward(x);
    cout << "y: rows=" << y.rows() << ", cols=" << y.cols() << endl;

    // 4) Simple loss: sum of all entries in y
    Tensor loss = sum(y);

    // 5) Backward pass
    backward(loss);

    // 6) Inspect some gradients

    cout << "\nSome gradients on x:" << endl;
    for (std::size_t i = 0; i < T; ++i) {
        cout << "row " << i << ": ";
        for (std::size_t j = 0; j < d_model; ++j) {
            double g = x.grad().data[i * d_model + j];
            cout << g << " ";
        }
        cout << endl;
    }

    cout << "\nSome gradients on W_q:" << endl;
    std::size_t Wq_rows = attn.W_q.W.rows();
    std::size_t Wq_cols = attn.W_q.W.cols();
    for (std::size_t i = 0; i < std::min<std::size_t>(Wq_rows, 3); ++i) {
        cout << "row " << i << ": ";
        for (std::size_t j = 0; j < std::min<std::size_t>(Wq_cols, 4); ++j) {
            double g = attn.W_q.W.grad().data[i * Wq_cols + j];
            cout << g << " ";
        }
        cout << endl;
    }

    return 0;
}
