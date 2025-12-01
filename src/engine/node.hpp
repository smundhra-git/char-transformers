#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>

namespace engine {
    // Base Node for Autograd
    struct Node : std::enable_shared_from_this<Node> {
        std::vector<Tensor> inputs; 

        Node() = default;
        virtual ~Node() = default;

        // grad_output matches the shape of the tensor produced by this node
        virtual void backward(const std::vector<double>& grad_output) = 0;
    };
}
