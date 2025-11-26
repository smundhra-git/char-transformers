#pragma once 

#include "../engine/tensor.hpp"

using namespace std;

namespace nn {
    //config for a linear layer
    struct LinearConfig {
        size_t in_features;
        size_t out_features;
        //this are input and output dimensions
        //later we can add init scale, bias - on or off etc etc
    };

    //Linear layer: y = xW +b

    //shapes:
    // - W [in_feat * out_feat], b[1 * out_feat], x [bs * in_feat], y [ bs * out_feat]

    struct Linear {
        engine::Tensor W; //weights
        engine::Tensor b; //bias

        //construct a linear layer with rnadomly initialized params
        explicit Linear(const LinearConfig& cfg);

        //forward pass - computer y = xW + b
        engine::Tensor forward(engine::Tensor& x);
    };
} //namespace nn