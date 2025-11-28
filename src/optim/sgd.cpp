#include "sgd.hpp"

using namespace std; 

namespace optim {
    void sgd_step(const vector<engine::Tensor&>& params, double lr){
        for(engine::Tensor& p : params){
            if(!p.p ||!p.require_grad()) continue;

            //ensure grad has some shape as data
            if(p.grad().rows != p.data().rows || p.grad().cols != p.data().cols){
                continue; //could throw error here too?
            }

            size_t n = p.data().size();
            for(size_t i = 0; i < n; i++){
                p.data().data[i] -= lr * p.grad().data[i];
            }
        }
    }
}