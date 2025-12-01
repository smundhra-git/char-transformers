#pragma once
#include "../engine/tensor.hpp"
#include "../engine/ops.hpp"

namespace nn {
    struct LinearConfig {
        size_t in_features;
        size_t out_features;
        bool bias = true;
    };

    class Linear {
    public:
        engine::Tensor W; // (In, Out)
        engine::Tensor b; // (Out)

        Linear(const LinearConfig& cfg);
        
        engine::Tensor forward(const engine::Tensor& x);
    };
}
