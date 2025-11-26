#include "tensor.hpp"
#include "node.hpp"


#include <cassert>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>

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
        Tensor* t,
        unordered_set<Tensor*>& visited,
        vector<Tensor*>& topo){

            if(!t) return;
            if(visited.count(t)) return;
            visited.insert(t);

            if(t->grad_fn){
                for(Tensor* inp : t->grad_fn->inputs){
                    build_topo(inp, visited, topo);
                }
            }
            topo.push_back(t);
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

        //build topological order of nodes
        unordered_set<Tensor*> visited;
        vector<Tensor*> topo;
        cout << "1" << endl;
        build_topo(&loss, visited, topo);

        cout << "2" << endl;
        
        //reverse topo for backprop
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Tensor* t = *it;
            if (!t->grad_fn) continue;

            // t->grad is dL/d(output)
            t->grad_fn->backward(t->grad);
        }

    }
}