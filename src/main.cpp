#include <iostream>
#include <cassert>
#include <cmath>

#include "math/matrix.hpp"
#include "engine/tensor.hpp"
#include "engine/ops.hpp"

using namespace std;
using namespace math;
using namespace engine;

bool almost_equal(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

int main() {
    cout << "=== Autograd matmul + relu + sum test ===" << endl;

    // a: 2x3
    math::Matrix A_m(2, 3, 0.0);
    A_m.data = {
        -1.0, 2.0, 3.0,
         4.0, -5.0, 6.0
    };

    // b: 3x2
    math::Matrix B_m(3, 2, 0.0);
    B_m.data = {
        1.0,  2.0,
        3.0,  4.0,
        -1.0, 0.5
    };

    Tensor a(A_m, /*requires_grad=*/true);
    Tensor b(B_m, /*requires_grad=*/true);

    // y = relu(a * b)
    Tensor z = matmul(a, b);

    Tensor y = relu(z);

    // loss = sum(y)
    Tensor loss = sum(y);

    // Backprop
    backward(loss);

    cout << "2" << endl;

    cout << "loss.data = " << loss.data.data[0] << endl;

    cout << "a.grad:" << endl;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            double g = a.grad.data[i * a.cols() + j];
            cout << g << " ";
        }
        cout << endl;
    }

    cout << "b.grad:" << endl;
    for (std::size_t i = 0; i < b.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            double g = b.grad.data[i * b.cols() + j];
            cout << g << " ";
        }
        cout << endl;
    }

    cout << "Test ran successfully (no asserts yet, just inspection)." << endl;
    return 0;
}
