#pragma once
#include "transformer_block.hpp"
#include "embedding.hpp"
#include "positional_encoding.hpp"
#include <vector>

namespace nn {
    struct TransformerConfig {
        size_t vocab_size;
        size_t d_model;
        size_t block_size;
        size_t d_ff;
        size_t n_layer;
        size_t n_head;
    };

    class Transformer {
    public:
        Embedding tok_emb;
        PositionalEncoding pos_enc; // Replaces learnable pos_emb
        std::vector<Block> blocks;
        engine::Tensor ln_f_gamma;
        engine::Tensor ln_f_beta;
        Linear lm_head;
        
        size_t block_size;

        Transformer(const TransformerConfig& cfg);
        
        engine::Tensor forward(const std::vector<int>& idx, size_t Batch);
    };
}
