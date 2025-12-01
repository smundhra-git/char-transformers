#pragma once 

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>

namespace engine {

    struct Node; // Forward declaration

    using Shape = std::vector<size_t>;

    // Shared storage for data and gradients to support zero-copy views
    struct Storage {
        std::vector<double> data;
        std::vector<double> grad;
        
        Storage(size_t size) : data(size), grad(size, 0.0) {}
        Storage(const std::vector<double>& d) : data(d), grad(d.size(), 0.0) {}
    };

    struct TensorBody {
        std::shared_ptr<Storage> storage;
        size_t offset; // Offset into storage
        Shape shape;
        std::vector<size_t> strides;
        bool require_grad;
        std::shared_ptr<Node> grad_fn;

        TensorBody() : offset(0), require_grad(false), grad_fn(nullptr) {}
    };

    class Tensor {
    public:
        std::shared_ptr<TensorBody> p;

        Tensor();
        
        // Create from shape
        Tensor(const Shape& shape, bool require_grad = false);
        
        // Create from flat data + shape
        Tensor(const std::vector<double>& data, const Shape& shape, bool require_grad = false);

        // Create view (internal)
        Tensor(std::shared_ptr<Storage> storage, size_t offset, const Shape& shape, const std::vector<size_t>& strides, bool require_grad);

        // Accessors
        const Shape& shape() const { return p->shape; }
        size_t shape(size_t dim) const { return p->shape[dim]; }
        size_t dim() const { return p->shape.size(); }
        size_t numel() const;
        
        // Raw access is tricky with views, we expose underlying storage for ops
        // NOTE: Ops must handle strides!
        std::vector<double>& data() { return p->storage->data; }
        const std::vector<double>& data() const { return p->storage->data; }
        
        // Helper to get actual linear index for a contiguous tensor (throws if non-contiguous?)
        // For simple ops, we assume contiguous for now or handle strides.
        bool is_contiguous() const;
        
        // Make the tensor contiguous in memory (copies if needed)
        Tensor contiguous() const;

        std::vector<double>& grad() { return p->storage->grad; }
        const std::vector<double>& grad() const { return p->storage->grad; }

        bool require_grad() const { return p->require_grad; }
        std::shared_ptr<Node>& grad_fn() { return p->grad_fn; }
        
        const std::vector<size_t>& strides() const { return p->strides; }
        size_t offset() const { return p->offset; }

        void zero_grad();

        // Helpers
        static Tensor zeros(const Shape& shape, bool require_grad = false);
        static Tensor constant(const Shape& shape, double value, bool require_grad = false);
        static Tensor randn(const Shape& shape, double mean = 0.0, double std = 1.0, bool require_grad = false);
        static Tensor kaiming_uniform(const Shape& shape, bool require_grad = false);

        // Views/Reshape (Zero Copy)
        Tensor reshape(const Shape& new_shape) const; 
        Tensor view(const Shape& new_shape) const; // Alias for reshape
    };

    void backward(Tensor& loss);
    
    // Debug printer
    std::ostream& operator<<(std::ostream& os, const Tensor& t);
}
