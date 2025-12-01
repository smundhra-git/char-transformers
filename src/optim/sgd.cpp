#include "sgd.hpp"

using namespace std; 

namespace optim {
    void sgd_step(const vector<engine::Tensor>& params, double lr){
        for(auto p : params){ // Copy handle
            if(!p.p || !p.require_grad()) continue;

            // Simple check: grad size must match data size
            if(p.grad().size() != p.data().size()){
                continue; 
            }

            auto& w = p.data();
            auto& g = p.grad();

            size_t n = w.size();
            for(size_t i = 0; i < n; i++){
                w[i] -= lr * g[i];
            }
        }
    }
}
