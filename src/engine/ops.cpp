#include "ops.hpp"
#include "../math/matrix.hpp"

#include <stdexcept>

namespace engine {
    using namespace std;
    //concrete Node for addition : out = a + b
    struct AddNode : public Node {
        //we do not need extra fields; inputs and outputs are enough

        void backward(const math::Matrix& grad_out) override {

            //gradient wrt each input is just grad_out (elementwise)
            for(Tensor& inp : inputs){
                if(!inp.require_grad()) continue;

                //ensure input->grad shape matches
                inp.grad().resize(grad_out.rows, grad_out.cols, 0.0);

                //accumulate : inp->grad += grad_out
                for(size_t i = 0; i < grad_out.size(); i++){
                    inp.grad().data[i] += grad_out.data[i];
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
        math::Matrix out_data = math::add(a.data(), b.data());

        bool requires = a.require_grad() || b.require_grad();
        Tensor out(out_data, requires);

        if(requires) {
            auto node = make_shared<AddNode>();
            node->inputs = {a, b};
            out.p->grad_fn = node;
        }
        return out;
    }

    struct SumNode : public Node {
        void backward(const math::Matrix& grad_out) override {
            //grad_out is scalar 1x1
            double g = grad_out.data[0];
            
            Tensor& inp = inputs[0];
            if(!inp.require_grad()) return;

            //dL/d(inp) = g* ones_like(inp)
            inp.grad().resize(inp.rows(),inp.cols(), 0.0);
            for(size_t i = 0; i<inp.grad().size(); i++){
                inp.grad().data[i] +=g;
            }
        }
    };

    Tensor sum(const Tensor& a){
        double s = 0.0;
        for(size_t i = 0; i<a.size(); i++){
            s += a.data().data[i];
        }

        math::Matrix m(1, 1, s);
        bool requires = a.require_grad();
        Tensor out(m, requires);

        if(requires) {
            auto node = make_shared<SumNode>();
            node->inputs = {a};
            out.p->grad_fn = node;
        }
        return out;
    }

    struct MatmulNode : public Node {
        //for matmul out = a*b
        //inputs[0] = a, inputs[1] = b

        void backward(const math::Matrix& grad_out) override {
            Tensor& a = inputs[0];
            Tensor& b = inputs[1];


            //dL/da = grad_out * b^T
            if(a.require_grad()){
                math::Matrix b_T = math::transpose(b.data());
                math::Matrix grad_a = math::matmul(grad_out, b_T);
                
                a.grad().resize(a.rows(), a.cols(), 0.0);
                for(size_t i = 0; i <a.grad().size(); i++){
                    a.grad().data[i] += grad_a.data[i];
                }
            }

            //dL/db = a^T * grad_out
            if(b.require_grad()) { 
                math::Matrix a_T = math::transpose(a.data());
                math::Matrix grad_b = math::matmul(a_T, grad_out);

                b.grad().resize(b.rows(), b.cols(), 0.0);
                for(size_t i = 0; i < b.grad().size();i++){
                    b.grad().data[i] += grad_b.data[i];
                }
            }

        }
    };

    Tensor matmul(const Tensor& a, const Tensor& b){
        if(a.cols()!=b.rows()) throw invalid_argument("incompataible rows and cols");

        math::Matrix out_data = math::matmul(a.data(), b.data());
        bool requires = a.require_grad() || b.require_grad();

        Tensor out(out_data, requires);

        if(requires){
            auto node = make_shared<MatmulNode>();
            node -> inputs = {a, b};
            out.p->grad_fn = node;
        }

        return out;
    }

    struct ReluNode : public Node {
        void backward(const math::Matrix& grad_out) override {
            Tensor& x = inputs[0];
            const math::Matrix& x_data = x.data();

            if(!x.require_grad()) return;

            x.grad().resize(x.rows(), x.cols(), 0.0);

            for(size_t i = 0; i<x.grad().size(); i++){
                double gate = x_data.data[i] > 0.0 ? 1.0 : 0.0;
                x.grad().data[i] += gate * grad_out.data[i];
            }
        }
    };

    Tensor relu(const Tensor& x){
        math::Matrix out_data(x.rows(), x.cols(), 0.0);
        //forward out = max(0,x)

        for(size_t i = 0; i< x.size(); i++){
            out_data.data[i] = max(0.0, x.data().data[i]);
        }

        bool requires = x.require_grad();
        Tensor out(out_data, requires);

        if(requires){
            auto node = make_shared<ReluNode>();
            node->inputs = {x};
            out.p->grad_fn = node;
        }
        
        return out;
    }

    struct BiasAddNode : public Node {
        void backward(const math::Matrix& grad_out) override {
            Tensor& x = inputs[0];
            Tensor& b = inputs[1];

            if(x.require_grad()) {
                x.grad().resize(grad_out.rows, grad_out.cols, 0.0);
                for(size_t i = 0; i < grad_out.size(); i++){
                    x.grad().data[i] += grad_out.data[i];
                }
            }

            if(b.require_grad()) {
                //b is [1 * out_feat]
                size_t batch = grad_out.rows;
                size_t out_features = grad_out.cols;
                b.grad().resize(1, out_features, 0.0);

                for(size_t i = 0; i<batch; i++){
                    for(size_t j = 0; j < out_features; j++){
                        double g = grad_out.data[i*out_features + j];
                        b.grad().data[j] += g;
                    }
                }
            }

        }
    };

    Tensor bias_add(const Tensor& x, const Tensor& b){
        if (b.rows() != 1 || x.cols() != b.cols()){
            throw invalid_argument("bias_add argument size error");
        }
        size_t batch = x.rows();
        size_t out_features = x.cols();

        //forward y
        math::Matrix out_data(batch, out_features, 0.0);

        for(size_t i = 0; i<batch; i++){
            for(size_t j = 0; j<out_features; j++){
                double xv = x.data().data[i*out_features + j];
                double bv = b.data().data[j];
                out_data.data[i * out_features + j] = xv + bv;
            }
        }
        bool requires = x.require_grad() || b.require_grad();
        Tensor out(out_data, requires);
        

        if(requires){
            auto node = make_shared<BiasAddNode>();
            node->inputs = {x, b};
            out.p->grad_fn = node;
        }
        return out;
    }

    struct MSELossNode : public Node {
        //inputs[0] = pred;
        //inputs[1] == target (usually require_grad = false)
        
        void backward(const math::Matrix& grad_out) override {
            Tensor& pred = inputs[0];
            Tensor& target = inputs[1];

            if(!pred.require_grad()) return;

            //grad_out is scalar 
            double g = grad_out.data[1];

            size_t n = pred.size();
            if(n == 0) return;

            pred.grad().resize(pred.rows(), pred.cols(), 0.0);

            for(size_t i = 0; i< n; i++){
                double diff = pred.data().data[i] - target.data().data[i];
            // d/d pred : 2/N * diff * g
                pred.grad().data[i] += (2.0/static_cast<double>(n)) * diff * g;
            }

            //for right now we will not propogate grad into target
        }
    };

    Tensor mse_loss(Tensor& pred, Tensor& target){
        if(pred.rows() != target.rows() || pred.cols() != target.cols()){
            throw invalid_argument ("mse_loss = size do not match");
        }

        double s = 0.0;
        size_t n = pred.size();

        for(size_t i = 0; i<n; i++){
            double diff = pred.data().data[i] - target.data().data[i];
            s += diff * diff;
        }

        double mse = ( n > 0) ? (s / static_cast<double>(n)) : 0.0;

        math::Matrix m(1, 1, mse);

        bool requires = pred.require_grad();

        Tensor out(m, requires);

        if(requires){
            auto node = make_shared<MSELossNode>();
            node->inputs = {pred, target};
            out.p->grad_fn = node;
        }

        return out;
    }


        struct SubNode : public Node {
            // out = a - b
            void backward(const math::Matrix& grad_out) override {
                Tensor& a = inputs[0];
                Tensor& b = inputs[1];
    
                if (a.require_grad()) {
                    a.grad().resize(grad_out.rows, grad_out.cols, 0.0);
                    for (std::size_t i = 0; i < grad_out.size(); ++i) {
                        a.grad().data[i] += grad_out.data[i];      // d/d a: +1
                    }
                }
    
                if (b.require_grad()) {
                    b.grad().resize(grad_out.rows, grad_out.cols, 0.0);
                    for (std::size_t i = 0; i < grad_out.size(); ++i) {
                        b.grad().data[i] -= grad_out.data[i];      // d/d b: -1
                    }
                }
            }
        };
    
        Tensor sub(const Tensor& a, const Tensor& b) {
            if (a.rows() != b.rows() || a.cols() != b.cols()) {
                throw std::invalid_argument("sub: shape mismatch");
            }
    
            math::Matrix out_data(a.rows(), a.cols(), 0.0);
            for (std::size_t i = 0; i < a.size(); ++i) {
                out_data.data[i] = a.data().data[i] - b.data().data[i];
            }
    
            bool requires = a.require_grad() || b.require_grad();
            Tensor out(out_data, requires);
            if (requires) {
                auto node = std::make_shared<SubNode>();
                node->inputs = { a, b };
                out.p->grad_fn = node;
            }
            return out;
        }
    

    struct HadamardNode : public Node {
        // out = a * b  (elementwise)
        void backward(const math::Matrix& grad_out) override {
            Tensor& a = inputs[0];
            Tensor& b = inputs[1];

            if (a.require_grad()) {
                a.grad().resize(grad_out.rows, grad_out.cols, 0.0);
                for (std::size_t i = 0; i < grad_out.size(); ++i) {
                    double d = b.data().data[i];  // ∂(a*b)/∂a = b
                    a.grad().data[i] += grad_out.data[i] * d;
                }
            }

            if (b.require_grad()) {
                b.grad().resize(grad_out.rows, grad_out.cols, 0.0);
                for (std::size_t i = 0; i < grad_out.size(); ++i) {
                    double d = a.data().data[i];  // ∂(a*b)/∂b = a
                    b.grad().data[i] += grad_out.data[i] * d;
                }
            }
        }
    };

    Tensor hadamard(const Tensor& a, const Tensor& b) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw std::invalid_argument("hadamard: shape mismatch");
        }

        math::Matrix out_data(a.rows(), a.cols(), 0.0);
        for (std::size_t i = 0; i < a.size(); ++i) {
            out_data.data[i] = a.data().data[i] * b.data().data[i];
        }

        bool requires = a.require_grad() || b.require_grad();
        Tensor out(out_data, requires);
        if (requires) {
            auto node = std::make_shared<HadamardNode>();
            node->inputs = { a, b };
            out.p->grad_fn = node;
        }
        return out;
    }


        struct ScaleNode : public Node {
            double alpha;
            explicit ScaleNode(double alpha_) : alpha(alpha_) {}
    
            void backward(const math::Matrix& grad_out) override {
                Tensor& a = inputs[0];
                if (!a.require_grad()) return;
    
                a.grad().resize(grad_out.rows, grad_out.cols, 0.0);
                for (std::size_t i = 0; i < grad_out.size(); ++i) {
                    a.grad().data[i] += alpha * grad_out.data[i];
                }
            }
        };
    
        Tensor scale(const Tensor& a, double alpha) {
            math::Matrix out_data(a.rows(), a.cols(), 0.0);
            for (std::size_t i = 0; i < a.size(); ++i) {
                out_data.data[i] = alpha * a.data().data[i];
            }
    
            bool requires = a.require_grad();
            Tensor out(out_data, requires);
            if (requires) {
                auto node = std::make_shared<ScaleNode>(alpha);
                node->inputs = { a };
                out.p->grad_fn = node;
            }
            return out;
        }


        struct SoftmaxRowNode : public Node {
            void backward(const math::Matrix& grad_out) override {
                Tensor& x = inputs[0];
                
                if(!x.require_grad()) return;
                
                const math::Matrix& y_data = math::row_softmax(x.data());
                size_t rows = grad_out.rows;
                size_t cols = grad_out.cols;

                if(rows !=y_data.rows || cols != y_data.cols) throw runtime_error("Softmaxrownode - backward shape missing");

                x.grad().resize(rows, cols, 0.0);

                for(size_t i = 0; i<rows; i++){
                    //computer for = sum_j g_j * y_j
                    double dot = 0.0;
                    for(size_t j = 0; j <cols; j++){
                        double g = grad_out.data[i * cols + j];
                        double yv = y_data.data[i * cols + j];
                        dot += g * yv;
                    }

                    //now for each j : DL/dx_j = y_j + (g_j - dot)
                    for(size_t j = 0; j < cols; j++){
                        double g = grad_out.data[i*cols + j];
                        double yv = y_data.data[i * cols + j];
                        double dz = yv * (g - dot);
                        x.grad().data[i * cols + j] += dz;
                    }
                }
            }
        };

        Tensor softmax_row(const Tensor& x){
            
            //forward delegate to math::row _softmax
            math::Matrix out_data = math::row_softmax(x.data());

            bool requires = x.require_grad();
            Tensor out(out_data, requires);

            if(requires){
                auto node = make_shared<SoftmaxRowNode>();
                node -> inputs = {x};

                out.p->grad_fn = node;

            }
            return out;

        }

        
    // ---------- Transpose (2D) ----------

    struct TransposeNode : public Node {

        void backward(const math::Matrix& grad_out) override {
            Tensor& x = inputs[0];
            if (!x.require_grad()) return;

            std::size_t in_rows = x.rows();
            std::size_t in_cols = x.cols();

            // grad_out should be [in_cols x in_rows]
            if (grad_out.rows != in_cols || grad_out.cols != in_rows) {
                throw std::runtime_error("TransposeNode::backward: shape mismatch");
            }

            x.grad().resize(in_rows, in_cols, 0.0);

            // dL/dx(i,j) = grad_out(j,i)
            for (std::size_t i = 0; i < in_rows; ++i) {
                for (std::size_t j = 0; j < in_cols; ++j) {
                    x.grad().data[i * in_cols + j] +=
                        grad_out.data[j * in_rows + i];
                }
            }
        }
    };

    Tensor transpose(const Tensor& x) {
        math::Matrix out_data = math::transpose(x.data()); // [cols x rows]

        bool requires = x.require_grad();
        Tensor out(out_data, requires);

        if (requires) {
            auto node = std::make_shared<TransposeNode>();
            node->inputs = { x};
            out.p->grad_fn = node;
        }

        return out;
    }

    

} //namespace engine