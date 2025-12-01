#include "tensor.hpp"
#include "node.hpp"

#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>
#include <unordered_set> // Fixed: Added this
#include <cmath>

namespace engine {

    // Utils
    static std::vector<size_t> compute_strides(const Shape& shape) {
        std::vector<size_t> strides(shape.size());
        size_t s = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shape[i];
        }
        return strides;
    }

    Tensor::Tensor() : p(std::make_shared<TensorBody>()) {}

    Tensor::Tensor(const Shape& shape, bool require_grad_) {
        p = std::make_shared<TensorBody>();
        p->shape = shape;
        p->strides = compute_strides(shape);
        
        size_t size = 1;
        for(auto s : shape) size *= s;
        
        p->data.resize(size, 0.0);
        p->grad.resize(size, 0.0);
        p->require_grad = require_grad_;
    }

    Tensor::Tensor(const std::vector<double>& data, const Shape& shape, bool require_grad_) {
        p = std::make_shared<TensorBody>();
        p->shape = shape;
        p->strides = compute_strides(shape);
        p->data = data;
        
        // Check size
        size_t size = 1;
        for(auto s : shape) size *= s;
        if (data.size() != size) {
            throw std::runtime_error("Tensor init: data size mismatch with shape");
        }

        p->grad.resize(size, 0.0);
        p->require_grad = require_grad_;
    }

    void Tensor::zero_grad() {
        std::fill(p->grad.begin(), p->grad.end(), 0.0);
    }

    Tensor Tensor::zeros(const Shape& shape, bool require_grad) {
        return Tensor(shape, require_grad);
    }

    Tensor Tensor::constant(const Shape& shape, double value, bool require_grad) {
        Tensor t(shape, require_grad);
        std::fill(t.data().begin(), t.data().end(), value);
        return t;
    }

    Tensor Tensor::randn(const Shape& shape, double mean, double std, bool require_grad) {
        Tensor t(shape, require_grad);
        static std::mt19937 gen(1337);
        std::normal_distribution<> d(mean, std);
        for(auto& v : t.data()) v = d(gen);
        return t;
    }
    
    Tensor Tensor::kaiming_uniform(const Shape& shape, bool require_grad) {
        // Kaiming/He initialization
        // Assumes shape is [fan_out, fan_in] or similar
        Tensor t(shape, require_grad);
        if (shape.size() < 2) return t; // fallback
        
        double fan_in = (double)shape[1]; // typical for Linear [out, in]
        double bound = std::sqrt(6.0 / fan_in); // for Uniform
        
        static std::mt19937 gen(42);
        std::uniform_real_distribution<> d(-bound, bound);
        for(auto& v : t.data()) v = d(gen);
        return t;
    }

    Tensor Tensor::reshape(const Shape& new_shape) const {
        size_t current_size = numel();
        size_t new_size = 1;
        for(auto s : new_shape) new_size *= s;
        
        if(current_size != new_size) throw std::runtime_error("Reshape size mismatch");
        
        Tensor out(p->data, new_shape, p->require_grad);
        return out;
    }

    // Topo sort
    using VisitedSet = std::unordered_set<TensorBody*>;
    using TopoList = std::vector<Tensor>;

    static void build_topo(Tensor& t, VisitedSet& visited, TopoList& topo) {
        if (!t.p) return;
        if (visited.count(t.p.get())) return;
        visited.insert(t.p.get());

        if (t.grad_fn()) {
            for (auto& input : t.grad_fn()->inputs) {
                build_topo(input, visited, topo);
            }
        }
        topo.push_back(t);
    }

    void backward(Tensor& loss) {
        if (!loss.require_grad()) return;
        
        // Scalar check
        if (loss.numel() != 1) throw std::runtime_error("Backward on non-scalar tensor");

        loss.zero_grad();
        loss.grad()[0] = 1.0;

        VisitedSet visited;
        TopoList topo;
        build_topo(loss, visited, topo);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Tensor& t = *it;
            if (t.grad_fn()) {
                t.grad_fn()->backward(t.grad());
            }
        }
    }

    std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << "Tensor(";
        os << "Shape=[";
        for(auto s : t.shape()) os << s << " ";
        os << "])";
        return os;
    }
}
