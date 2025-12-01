#pragma once
#include "../engine/tensor.hpp"
#include <vector>
#include <unordered_map>

namespace optim {
    
    class Adam {
    public:
        Adam(std::vector<engine::Tensor>& params, double lr=1e-3, double beta1=0.9, double beta2=0.999, double eps=1e-8);
        
        void step();
        void zero_grad();

    private:
        struct State {
            std::vector<double> m;
            std::vector<double> v;
        };
        
        std::vector<engine::Tensor> params;
        double lr;
        double beta1;
        double beta2;
        double eps;
        int t; // Time step
        
        // Key: TensorBody pointer address
        std::unordered_map<void*, State> states;
    };

}

