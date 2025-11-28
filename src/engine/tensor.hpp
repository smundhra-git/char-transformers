#pragma once 

#include "../math/matrix.hpp"
#include <memory>
#include <vector>

using namespace std;

namespace engine {
    struct Node; // Forward declaration

    struct TensorBody {
        math::Matrix data;
        math::Matrix grad;
        bool require_grad;
        shared_ptr<Node> grad_fn;

        TensorBody(): data(), grad(), require_grad(false), grad_fn(nullptr) {}
    };

    //tensor : wraps a math::Matrix and autograd metada
    //for now we will keep this as 2x2 only, which is fine for our use case, later we can change this? Ofcourse we can
    //TODO:: UPDATE THIS TO N-Dim

    class Tensor {
        public :
            shared_ptr<TensorBody> p;

            Tensor() : p(make_shared<TensorBody>()) {};
            Tensor(const math::Matrix& values, bool require_grad_){
                p = make_shared<TensorBody>();
                p->data = values;
                p->grad = math::Matrix(values.rows, values.cols, 0.0);
                p->require_grad = require_grad_;
                p->grad_fn = nullptr;
            }

            //accessors 
            size_t rows() const { return p->data.rows; }
            size_t cols() const { return p->data.cols;}
            size_t size() const { return p->data.size(); }

            math::Matrix& data() {return p->data;}

            const math::Matrix& data() const { return p->data; }

            math::Matrix& grad() {return p->grad;}

            const math::Matrix& grad() const { return p->grad; }
            
            bool require_grad() const {return p->require_grad;}

            void zero_grad(){
                p->grad.resize(p->data.rows, p->data.cols, 0.0);
                p->grad.fill(0.0);
            }
    };

    void backward(Tensor& loss);
}