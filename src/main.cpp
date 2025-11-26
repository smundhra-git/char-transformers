#include <iostream>
#include "math/matrix.hpp"
#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "nn/linear.hpp"

using namespace std;
using namespace math;
using namespace engine;
using namespace nn;

int main() {
    cout << "=== Linear + bias (fused) test ===" << endl;

    LinearConfig cfg{3, 2};
    Linear layer(cfg);

    // Manually set W: [3 x 2]
    layer.W.data.data = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };

    // Manually set b: [1 x 2]
    layer.b.data.data = { 0.5, -1.0 };

    // Input x: [2 x 3]
    Matrix X_m(2, 3, 0.0);
    X_m.data = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    Tensor x(X_m, /*require_grad=*/false);

    Tensor y = layer.forward(x);

    cout << "y (forward):" << endl;
    for (size_t i = 0; i < y.rows(); ++i) {
        for (size_t j = 0; j < y.cols(); ++j) {
            cout << y.data.data[i * y.cols() + j] << " ";
        }
        cout << endl;
    }

    Tensor loss = sum(y);
    backward(loss);

    cout << "\nloss = " << loss.data.data[0] << endl;

    cout << "\nW.grad:" << endl;
    for (size_t i = 0; i < layer.W.rows(); ++i) {
        for (size_t j = 0; j < layer.W.cols(); ++j) {
            cout << layer.W.grad.data[i * layer.W.cols() + j] << " ";
        }
        cout << endl;
    }

    cout << "\nb.grad:" << endl;
    for (size_t j = 0; j < layer.b.cols(); ++j) {
        cout << layer.b.grad.data[j] << " ";
    }
    cout << endl;

    return 0;
}
