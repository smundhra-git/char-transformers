#pragma once 

#include "../engine/tensor.hpp"
#include "embedding.hpp"
#include "transformer_block.hpp"
#include "linear.hpp"
#include "vector"

using namespace std;

namespace nn {

    struct CharTransformerConfig {
        size_t vocab_size;
        size_t d_model;
        size_t d_ff;
        bool casual;
    };

    class CharTransformer {
        public:
            Embedding emb; //vocab_size -> d_model
            TransformerBlock block; //d_model -> d_model
            Linear lm_head; //d_model -> vocab_size

            explicit CharTransformer(const CharTransformerConfig& cfg);

            //forward ids [T] -> logits [T * vocab_size]
            engine::Tensor forward(const vector<int>& ids);
    };
}