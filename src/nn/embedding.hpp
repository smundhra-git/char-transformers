#pragma once
#include "../engine/tensor.hpp"
#include "../engine/ops.hpp"

namespace nn {
    struct EmbeddingConfig {
        size_t vocab_size;
        size_t d_model;
    };

    class Embedding {
    public:
        engine::Tensor W;

        Embedding(const EmbeddingConfig& cfg);
        
        // Forward takes raw indices vector (flattened batch)
        // Returns (Batch, T, D)
        // We pass shape explicitly: Batch, T
        engine::Tensor forward(const std::vector<int>& token_ids, size_t Batch, size_t T);
    };
}
