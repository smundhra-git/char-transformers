#pragma once
#include "../engine/tensor.hpp"
#include "linear.hpp"
#include <vector>

namespace nn {
    struct SelfAttentionConfig {
        size_t d_model;
        size_t n_head;
        size_t block_size;
        bool causal;
    };
    
    // KV Cache structure
    struct KVCache {
        engine::Tensor k;
        engine::Tensor v;
        size_t current_len = 0; // Tracks how many tokens are cached
        
        // Clears cache (useful at start of generation)
        void clear() {
            current_len = 0;
            // We don't free memory, just reset cursor
            // Or we could release tensors.
            k = engine::Tensor();
            v = engine::Tensor();
        }
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
        
        // Forward with optional KV Cache
        engine::Tensor forward(const engine::Tensor& x, KVCache* cache = nullptr);
    };
}
