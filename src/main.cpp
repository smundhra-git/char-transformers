#include <iostream>
#include <vector>
#include <cstddef>
#include <algorithm>

#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "nn/char_transformer.hpp"
#include "optim/sgd.hpp"

using namespace std;
using namespace engine;
using namespace nn;

int main() {
    cout << "=== CharTransformer tiny training test ===" << endl;

    // ----------------------------------------------------
    // 1. Hyperparameters & fake "vocab"
    // ----------------------------------------------------
    std::size_t vocab_size = 8;   // pretend we have 8 chars in vocab
    std::size_t d_model    = 8;
    std::size_t d_ff       = 16;
    bool causal            = true;

    CharTransformerConfig cfg{
        vocab_size,
        d_model,
        d_ff,
        causal
    };

    CharTransformer model(cfg);

    // Tiny toy "sequence": length T = 4
    std::size_t T = 4;
    std::vector<int> x_ids = {0, 1, 2, 3}; // input tokens
    std::vector<int> y_ids = {1, 2, 3, 4}; // target: "next token"

    // ----------------------------------------------------
    // 2. Collect parameters for SGD as Tensor handles
    //    (copies share same underlying TensorBody)
    // ----------------------------------------------------
    std::vector<Tensor> params;

    // Embedding weights
    params.push_back(model.emb.W);

    // Self-attention weights inside the block
    params.push_back(model.block.sa.W_q.W);
    params.push_back(model.block.sa.W_k.W);
    params.push_back(model.block.sa.W_v.W);
    params.push_back(model.block.sa.W_o.W);

    // Feed-forward weights + biases
    params.push_back(model.block.ff1.W);
    params.push_back(model.block.ff1.b);
    params.push_back(model.block.ff2.W);
    params.push_back(model.block.ff2.b);

    // LM head weights + bias
    params.push_back(model.lm_head.W);
    params.push_back(model.lm_head.b);

    double lr = 0.1;

    // ----------------------------------------------------
    // 3. Single forward/backward sanity test
    // ----------------------------------------------------
    {
        Tensor logits = model.forward(x_ids); // [T x vocab_size]
        cout << "logits: rows=" << logits.rows()
             << ", cols=" << logits.cols() << endl;

        Tensor loss = cross_entropy_logits(logits, y_ids);
        cout << "Initial loss (no training) = "
             << loss.data().data[0] << endl;

        // Zero all grads
        for (auto& p : params) {
            if (p.require_grad()) p.zero_grad();
        }

        backward(loss);

        cout << "\nSome gradients on embedding W (row 0):" << endl;
        std::size_t emb_cols = model.emb.W.cols();
        cout << "emb.W.grad row 0: ";
        for (std::size_t j = 0; j < std::min<std::size_t>(emb_cols, 8); ++j) {
            double g = model.emb.W.grad().data[j];
            cout << g << " ";
        }
        cout << "\n" << endl;
    }

    // ----------------------------------------------------
    // 4. Tiny training loop: watch loss decrease
    // ----------------------------------------------------
    int steps = 1000;
    for (int step = 0; step < steps; ++step) {
        // Forward
        Tensor logits = model.forward(x_ids);     // [T x vocab_size]
        Tensor loss   = cross_entropy_logits(logits, y_ids); // scalar

        if (step % 100 == 0) {
            cout << "step " << step
                 << "  loss = " << loss.data().data[0] << endl;
        }

        // Zero grads
        for (auto& p : params) {
            if (p.require_grad()) p.zero_grad();
        }

        // Backprop
        backward(loss);

        // SGD update
        optim::sgd_step(params, lr);
    }

    // Final check
    {
        Tensor logits = model.forward(x_ids);
        Tensor loss   = cross_entropy_logits(logits, y_ids);
        cout << "\nFinal loss after training = "
             << loss.data().data[0] << endl;
    }

    return 0;
}
