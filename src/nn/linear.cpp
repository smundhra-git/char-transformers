#include "nn/linear.hpp"

#include "engine/ops.hpp"
#include "math/matrix.hpp"

#include <random>

using namespace std;

namespace nn {

    using engine::Tensor;
    using math::Matrix;

    // Autograd node for y = xW + b
    struct LinearNode : public engine::Node {
        // We will store pointers to the actual parameter/input tensors:
        // inputs[0] = x
        // inputs[1] = W
        // inputs[2] = b

        void backward(const math::Matrix& grad_out) override {
            Tensor& x = inputs[0];
            Tensor& W = inputs[1];
            Tensor& b = inputs[2];

            // Shapes:
            // x: [batch x in_features]
            // W: [in_features x out_features]
            // b: [1 x out_features]
            // grad_out: [batch x out_features]

            std::size_t batch = grad_out.rows;
            std::size_t in_features = x.cols();
            std::size_t out_features = W.cols();

            // dL/dx = grad_out * W^T
            if (x.require_grad()) {
                math::Matrix W_T = math::transpose(W.data());
                math::Matrix grad_x = math::matmul(grad_out, W_T);

                x.grad().resize(batch, in_features, 0.0);
                for (std::size_t i = 0; i < x.grad().size(); ++i) {
                    x.grad().data[i] += grad_x.data[i];
                }
            }

            // dL/dW = x^T * grad_out
            if (W.require_grad()) {
                math::Matrix x_T = math::transpose(x.data());
                math::Matrix grad_W = math::matmul(x_T, grad_out);

                W.grad().resize(in_features, out_features, 0.0);
                for (std::size_t i = 0; i < W.grad().size(); ++i) {
                    W.grad().data[i] += grad_W.data[i];
                }
            }

            // dL/db = sum over rows of grad_out
            if (b.require_grad()) {
                b.grad().resize(1, out_features, 0.0);

                for (std::size_t i = 0; i < batch; ++i) {
                    for (std::size_t j = 0; j < out_features; ++j) {
                        double g = grad_out.data[i * out_features + j];
                        b.grad().data[j] += g;
                    }
                }
            }
        }
    };

    Linear::Linear(const LinearConfig& cfg)
        : W(), b() {

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distrib(-0.1, 0.1);

        // W: [in_features x out_features]
        Matrix w_matrix(cfg.in_features, cfg.out_features, 0.0);
        for (std::size_t i = 0; i < w_matrix.size(); ++i) {
            w_matrix.data[i] = distrib(gen);
        }
        W = Tensor(w_matrix, /*require_grad=*/true);

        // b: [1 x out_features]
        Matrix b_matrix(1, cfg.out_features, 0.0);
        for (std::size_t i = 0; i < b_matrix.size(); ++i) {
            b_matrix.data[i] = distrib(gen);
        }
        b = Tensor(b_matrix, /*require_grad=*/true);
    }

    Tensor Linear::forward(Tensor& x) {
        // x: [batch x in_features]
        // W: [in_features x out_features]
        // b: [1 x out_features]

        std::size_t batch = x.rows();
        std::size_t in_features = x.cols();
        std::size_t out_features = W.cols();

        // 1. Compute y_data = xW + b (matrix-level)
        math::Matrix y_data = math::matmul(x.data(), W.data()); // [batch x out_features]

        for (std::size_t i = 0; i < batch; ++i) {
            for (std::size_t j = 0; j < out_features; ++j) {
                y_data.data[i * out_features + j] += b.data().data[j];
            }
        }

        // 2. Wrap into a Tensor with autograd
        bool requires = x.require_grad() || W.require_grad() || b.require_grad();
        Tensor y(y_data, requires);

        if (requires) {
            auto node = std::make_shared<LinearNode>();
            // store pointers to existing, long-lived tensors
            node->inputs = { x, W, b };
            y.p->grad_fn = node;
        }

        return y;
    }

} // namespace nn
