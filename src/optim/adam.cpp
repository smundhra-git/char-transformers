#include "adam.hpp"
#include <cmath>

namespace optim {
    using namespace engine;

    Adam::Adam(std::vector<Tensor>& p, double lr, double beta1, double beta2, double eps)
        : params(p), lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0) 
    {
        // Initialize states for all params
        for(auto& param : params) {
            if(param.require_grad()) {
                size_t n = param.numel();
                void* key = param.p.get();
                states[key] = {
                    std::vector<double>(n, 0.0), // m
                    std::vector<double>(n, 0.0)  // v
                };
            }
        }
    }

    void Adam::zero_grad() {
        for(auto& param : params) {
            if(param.require_grad()) {
                param.zero_grad();
            }
        }
    }

    void Adam::step() {
        t++;
        
        // Bias correction terms
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);

        for(auto& param : params) {
            if(!param.require_grad()) continue;
            if(param.grad().empty()) continue;

            void* key = param.p.get();
            if(states.find(key) == states.end()) continue;

            State& s = states[key];
            
            std::vector<double>& m = s.m;
            std::vector<double>& v = s.v;
            std::vector<double>& w = param.data();
            const std::vector<double>& g = param.grad();
            
            size_t n = param.numel();

            for(size_t i=0; i<n; ++i) {
                double grad = g[i];

                // Update moments
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;

                // Bias correction
                double m_hat = m[i] / bias_correction1;
                double v_hat = v[i] / bias_correction2;

                // Update weight
                w[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
}

