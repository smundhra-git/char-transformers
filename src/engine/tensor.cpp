#include "tensor.hpp"
#include "node.hpp"

#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <iostream>

namespace engine {

    using std::runtime_error;
    using std::unordered_set;
    using std::vector;

    // ==========================
    // Autograd: topo build + backward
    // ==========================

    // helper DFS to build topological order of the nodes
    static void build_topo(
        Tensor& t,
        unordered_set<TensorBody*>& visited,
        vector<Tensor>& topo)
    {
        // identify this Tensor by its underlying storage pointer
        TensorBody* key = t.p.get();
        if (!key) return;
        if (visited.count(key)) return;
        visited.insert(key);

        // recurse on parents (inputs of grad_fn), if any
        if (t.p->grad_fn) {
            for (Tensor& inp : t.p->grad_fn->inputs) {
                build_topo(inp, visited, topo);
            }
        }

        // push this tensor after its dependencies
        topo.push_back(t);
    }

    // implement a real graph traversal here
    void backward(Tensor& loss) {
        // basic sanity check - we expect loss to be scalar (1x1) for now
        if (!loss.require_grad()) {
            // no gradients requested, nothing to do
            return;
        }

        if (loss.rows() != 1 || loss.cols() != 1) {
            throw runtime_error("backward() currently only supports scalar loss (1x1 Tensor)");
        }

        // initialize gradient of loss dL/dL = 1
        loss.zero_grad();
        if (loss.grad().size() == 0) {
            throw runtime_error("backward(): loss.grad has zero size after zero_grad");
        }
        loss.grad().data[0] = 1.0;

        // build topological order of nodes
        unordered_set<TensorBody*> visited;
        vector<Tensor> topo;
        build_topo(loss, visited, topo);

        // reverse topo for backprop
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Tensor& t = *it;
            if (!t.p->grad_fn) continue;

            // t.grad() is dL/d(output) for that node
            t.p->grad_fn->backward(t.grad());
        }
    }

} // namespace engine
