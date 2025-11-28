#pragma once

#include <vector>
#include "../engine/tensor.hpp"

using namespace std;

namespace optim {
    //inplace SGD update 
    //param.data -= lr * param.grad
    //assumes grad has been filled by backward()

    void sgd_step(const vector<engine::Tensor>& params, double lr);
}