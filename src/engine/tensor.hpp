#pragma once 

#include "../math/matrix.hpp"
#include <memory>
#include <vector>

using namespace std;

namespace engine {
    struct Node; //forward declaration of Node

    //tensor : wraps a math::Matrix and autograd metada
    //for now we will keep this as 2x2 only, which is fine for our use case, later we can change this? Ofcourse we can
    //TODO:: UPDATE THIS TO N-Dim

    struct Tensor{
        math::Matrix data; //actual numeric values
        math::Matrix grad; //gradients w.r.t tensor
        bool require_grad = false; //to track gradients

        //the node (operation) that will produce this tensor
        //nullptr if this is a leaf tensor (e.g. input data or a parameter created by user/me)
        
        shared_ptr<Node> grad_fn;

        //Constructor
        Tensor(); //default is empty and require_grad is false

        //construct from existing matrix
        Tensor(const math::Matrix& values, bool require_grad = false);

        //convience factory for zeroes
        static Tensor zeros(size_t rows, size_t cols, bool require_grad = false);

        //same for constant
        static Tensor constant(size_t rows, size_t cols, double value, bool require_grad = false);

        //shape helpers
        size_t rows() const;
        size_t cols() const;
        size_t size() const;

        //clear gradient
        void zero_grad();
    };

    //backward entry point :
    //Given scalar loss tensor(1x1) matrix, compute gradients for all
    //leaf tensors that have require_grad = true and are connected to this loss
    void backward(Tensor& loss);
} //namespace engine