#pragma once

#include <vector>
#include "../engine/tensor.hpp"

namespace nn {

    struct EmbeddingConfig {
        size_t vocab_size;
        size_t d_model;
    };

    //maps token ids to dense vectors.
    //input a vector<int> of token_ids of size N
    //output tensor of size N * d-model

    class Embedding{
        public :
            engine::Tensor W; //[vocab_size * d_model] trainable

            explicit Embedding(const EmbeddingConfig& cfg);

            //forward 
            //token_ids.size() = N, return Tensor [N*d_model]
            engine::Tensor forward(const vector<int>& token_ids);

    };

}   //namespace nn