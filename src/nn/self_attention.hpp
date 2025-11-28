#pragma once

#include "../engine/tensor.hpp"
#include "linear.hpp"

using namespace std;

namespace nn{
    struct SelfAttentionConfig {
        size_t d_model; //i/o dimesnion
        bool casual; //if true, apply casual mask (no loooking ahead)

    };

    //single head self-attention, batch_size = 1 for now
    class SelfAttention {
        public : 
            nn::Linear W_q;
            nn::Linear W_k;
            nn::Linear W_v;
            nn::Linear W_o;

            bool casual;

            explicit SelfAttention(const SelfAttentionConfig& cfg);

            //forward x : [T * d_model] return [T*d_model]

            engine::Tensor forward (engine::Tensor& x);
    };
}