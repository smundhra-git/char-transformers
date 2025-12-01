#pragma once
#include "../engine/tensor.hpp"
#include "self_attention.hpp"
#include "linear.hpp"

namespace nn {
    class MLP {
    public:
        Linear c_fc;
        Linear c_proj;
        
        MLP(size_t d_model, size_t d_ff);
        engine::Tensor forward(const engine::Tensor& x);
    };

    class Block {
    public:
        MultiHeadAttention sa;
        MLP ff;
        engine::Tensor ln1_gamma, ln1_beta;
        engine::Tensor ln2_gamma, ln2_beta;

        Block(const SelfAttentionConfig& cfg, size_t d_ff);
        
        // Forward with optional cache
        engine::Tensor forward(const engine::Tensor& x, KVCache* cache = nullptr);
    };
}
