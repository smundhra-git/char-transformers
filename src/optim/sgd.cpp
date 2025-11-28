#include "sgd.hpp"

using namespace std; 

namespace optim {
    void sgd_step(const vector<engine::Tensor>& params, double lr){
        for(auto p : params){
            if(!p.p ||!p.require_grad()) continue;

            //ensure grad has some shape as data
            if(p.grad().rows != p.data().rows || p.grad().cols != p.data().cols){
                continue; //could throw error here too?
            }

            auto& w = p.data();
            auto& g = p.grad();

            size_t n = w.size();
            for(size_t i = 0; i < n; i++){
                w.data[i] -= lr * g.data[i];
            }
        }
    }
}