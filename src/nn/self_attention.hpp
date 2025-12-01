#pragma once
#include "../engine/tensor.hpp"
#include "linear.hpp"

namespace nn {
    struct SelfAttentionConfig {
        size_t d_model;
        size_t n_head;
        size_t block_size;
        bool causal;
    };

    class MultiHeadAttention {
    public:
        Linear w_q;
        Linear w_k;
        Linear w_v;
        Linear w_o;
        
        size_t n_head;
        size_t d_model;
        bool causal;

        MultiHeadAttention(const SelfAttentionConfig& cfg);
        
        engine::Tensor forward(const engine::Tensor& x);
    };
}
