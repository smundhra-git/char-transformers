#pragma once

#include "../math/matrix.hpp"
#include <memory>
#include <vector>

namespace engine {
    struct Tensor; //forward declarion


    //node - base case for operations in the computation graph

    //each node will represent one "op" that produced an output tensor from one or more inputs 
    //During backpropagation we will call backwar() on each Node in reverse topological order

    struct Node : enable_shared_from_this<Node> {
        //pointer to input tensor. We do not own this; Tensor owns nodes via the shared_Ptr

        vector<Tensor*> inputs;

        //optional - we can track whether this node has been visited in a backward pass but for now we will just handle this in the backward() free function

        Node() = default;
        virtual ~Node() = default;

        //backward pass for this operation
        //grad_output : gradient of the loss wrt this node's output - will have the same shape as output
        //This implementation will 
        // 1. Computer grad_input for each input
        // 2. accumulate into input -> grad (since multiple nodes could contribute to the same Tensor)
        //use grad stored in output->grad
        virtual void backward(const math::Matrix& grad_output) = 0;
    };
} //namespace engine

