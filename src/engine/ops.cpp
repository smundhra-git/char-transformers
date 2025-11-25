#include "ops.hpp"
#include "../math/matrix.hpp"

#include "stdexcept"

namespace engine {
    using namespace std;
    //concrete Node for addition : out = a + b
    struct AddNode : public Node {
        //we do not need extra fields; inputs and outputs are enough

        void backward() override {
            //output -> grad is dL/d(out)

            const math::Matrix& grad_out = output->grad;

            //gradient wrt each input is just grad_out (elementwise)
            for(Tensor* inp : inputs){
                if(!inp -> require_grad) continue;

                //ensure input->grad shape matches
                inp->grad.resize(grad_out.rows, grad_out.cols, 0.0);

                //accumulate : inp->grad += grad_out
                for(size_t i = 0; i < grad_out.size(); i++){
                    inp->grad.data[i] += grad_out.data[i];
                }
            }
        }
    };

    Tensor add(const Tensor& a, const Tensor& b){
        //shape checks
        if(a.rows()!=b.rows() || a.cols()!=b.cols()){
            throw invalid_argument("Tensor add shapes different");
        }

        //forward numeric op
        math::Matrix out_data = math::add(a.data, b.data);

        bool requires = a.require_grad || b.require_grad;
        Tensor out(out_data, requires);

        if(requires) {
            auto node = make_shared<AddNode>();
            node->inputs = {const_cast<Tensor*>(&a), const_cast<Tensor*>(&b)};
            node->output = &out;
            out.grad_fn = node;
        }
        return out;
    }

    struct SumNode : public Node {
        void backward() override {
            //grad_out is scalar 1x1
            const math::Matrix& grad_out = output -> grad;
            double g = grad_out.data[0];
            
            Tensor* inp = inputs[0];
            if(!inp->require_grad) return;

            //dL/d(inp) = g* ones_like(inp)
            inp->grad.resize(inp->rows(),inp->cols(), 0.0);
            for(size_t i = 0; i<inp->grad.size(); i++){
                inp->grad.data[i] +=g;
            }
        }
    };

    Tensor sum(const Tensor& a){
        double s = 0.0;
        for(size_t i = 0; i<a.size(); i++){
            s += a.data.data[i];
        }

        math::Matrix m(1, 1, s);
        bool requires = a.require_grad;
        Tensor out(m, requires);

        if(requires) {
            auto node = make_shared<SumNode>();
            node->inputs = {const_cast<Tensor*>(&a)};
            node->output = &out;
            out.grad_fn = node;
        }
        return out;
    }
} //namespace engine