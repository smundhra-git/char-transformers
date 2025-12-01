#pragma once 

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>

namespace engine {

    struct Node; // Forward declaration

    using Shape = std::vector<size_t>;

    struct TensorBody {
        std::vector<double> data;
        std::vector<double> grad;
        Shape shape;
        std::vector<size_t> strides;
        bool require_grad;
        std::shared_ptr<Node> grad_fn;

        TensorBody() : require_grad(false), grad_fn(nullptr) {}
    };

    class Tensor {
    public:
        std::shared_ptr<TensorBody> p;

        Tensor();
        
        // Create from shape
        Tensor(const Shape& shape, bool require_grad = false);
        
        // Create from flat data + shape
        Tensor(const std::vector<double>& data, const Shape& shape, bool require_grad = false);

        // Accessors
        const Shape& shape() const { return p->shape; }
        size_t shape(size_t dim) const { return p->shape[dim]; }
        size_t dim() const { return p->shape.size(); }
        size_t numel() const { return p->data.size(); }
        
        std::vector<double>& data() { return p->data; }
        const std::vector<double>& data() const { return p->data; }

        std::vector<double>& grad() { return p->grad; }
        const std::vector<double>& grad() const { return p->grad; }

        bool require_grad() const { return p->require_grad; }
        std::shared_ptr<Node>& grad_fn() { return p->grad_fn; }

        void zero_grad();

        // Helpers
        static Tensor zeros(const Shape& shape, bool require_grad = false);
        static Tensor constant(const Shape& shape, double value, bool require_grad = false);
        static Tensor randn(const Shape& shape, double mean = 0.0, double std = 1.0, bool require_grad = false);
        static Tensor kaiming_uniform(const Shape& shape, bool require_grad = false);

        // Views/Reshape
        Tensor reshape(const Shape& new_shape) const; 
    };

    void backward(Tensor& loss);
    
    // Debug printer
    std::ostream& operator<<(std::ostream& os, const Tensor& t);
}
