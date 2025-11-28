#pragma once 

#include "tensor.hpp"
#include "node.hpp"
#include <vector>

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

    //mean squared error loss, same same pred and target return 1x1 scalar
    Tensor mse_loss(Tensor& pred, Tensor& target);

    //elemntwise a-b
    Tensor sub(const Tensor&a, const Tensor& b);

    //elementwise a*b
    Tensor hadamard(const Tensor& a, const Tensor& b);

    //alpha * a
    Tensor scale(const Tensor& a, double alpha);

    Tensor softmax_row(const Tensor& x);

    Tensor transpose(const Tensor& x);

    //cross entropy loss from logits _ int targets
    Tensor cross_entropy_logits(const Tensor& logits, const vector<int>& targets);


} //namespace engine