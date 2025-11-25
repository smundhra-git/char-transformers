#include "tensor.hpp"
#include "node.hpp"


#include <cassert>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace engine {
    using namespace std;
    Tensor::Tensor() : data(), grad(), require_grad(false), grad_fn(nullptr){
        //empty tensor
    }

    Tensor::Tensor(const math::Matrix& values, bool require_grad_)
    : data(values), grad(values.rows, values.cols, 0.0), require_grad(require_grad_), grad_fn(nullptr){

    }

    Tensor Tensor::zeros(size_t rows, size_t cols, bool require_grad_){
        math::Matrix m(rows, cols, 0.0);
        Tensor t(m, require_grad_);
        return t;
    }

    Tensor Tensor::constant(size_t rows, size_t cols, double value, bool require_grad_){
        math::Matrix m(rows, cols, value);
        Tensor t(m, require_grad_);
        return t;
    }

    size_t Tensor::rows() const {
        return data.rows;
    }

    size_t Tensor::cols() const {
        return data.cols;
    }

    size_t Tensor::size() const {
        return data.size();
    }

    void Tensor::zero_grad(){
        grad.resize(data.rows, data.cols, 0.0);
        grad.fill(0.0); //just to be explicit lowkey
    }

    //helper DFS to build topoplogical order of the nodes
    static void build_topo(
        const shared_ptr<Node>& node,
        unordered_set<Node*>& visited,
        vector<shared_ptr<Node>>& topo){

            if(!node) return;
            Node* raw = node.get();
            if(visited.count(raw)) return;
            visited.insert(raw);

            for(Tensor* inp : node->inputs){
                if(inp && inp->grad_fn){
                    build_topo(inp->grad_fn, visited, topo);
                }
            }
            topo.push_back(node);
        } 

    //implement a real graph traversal here
    void backward(Tensor& loss){
        //basic sanity check - we expect loss to be scalarr (1x1) for now
        if(!loss.require_grad){
            //no gradients found, nothing to do
            return;
        }

        if(loss.rows()!=1 || loss.cols() != 1){
            throw runtime_error("backward() currently is only scalar");
        }

        //initialize gradient of loss dL/dL = 1
        loss.zero_grad();
        loss.grad.data[0] = 1.0;

        //if there is no grad_fn, there is no grpah to traverse
        if(!loss.grad_fn){
            return;
        }

        //build topological order of nodes
        unordered_set<Node*> visited;
        vector<shared_ptr<Node>> topo;
        build_topo(loss.grad_fn, visited, topo);

        //traverse in reverse topopligical order
        for(auto it = topo.rbegin(); it != topo.rend(); it++){
            shared_ptr<Node> node = *it;
            node -> backward();
        }

    }
}