#pragma once

#include "self_attention.hpp"
#include "linear.hpp"
#include "../engine/tensor.hpp"

using namespace std;

namespace nn{

    struct TransformerBlockConfig{
        size_t d_model; //embeddgings dimension
        size_t d_ff; //hidden dimension in the MLP
        bool casual; //pass through to self attention
    };

    class TransformerBlock {
        public :
        SelfAttention sa; //self attention sublayer
        nn::Linear ff1; //first FF layer : d_model ->d_ff
        nn::Linear ff2; //pass through to selfattention

        explicit TransformerBlock(const TransformerBlockConfig& cfg);

        //forward 
        engine::Tensor forward(engine::Tensor& x);
    };
}