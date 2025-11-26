#pragma once 

#include "tensor.hpp"
#include "node.hpp"

namespace engine {
    //forward declaration of add op
    Tensor add(const Tensor& a, const Tensor& b);

    //sums all elements of a into a scalar (1x1) tensor
    Tensor sum(const Tensor& a);

    Tensor matmul(const Tensor& a, const Tensor& b);

    //ReLu activation
    Tensor relu(const Tensor& x);

    //y(i,j) = x(i, j) + b(0, j)
    Tensor bias_add(const Tensor& x, const Tensor& b);
} //namespace engine