#pragma once 

#include "tensor.hpp"
#include "node.hpp"

namespace engine {
    //forward declaration of add op
    Tensor add(const Tensor& a, const Tensor& b);

    //sums all elements of a into a scalar (1x1) tensor
    Tensor sum(const Tensor& a);
} //namespace engine