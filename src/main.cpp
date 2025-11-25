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
    cout << "=== Autograd add + sum test ===" << endl;

    // Create 2x2 leaf tensors a and b
    math::Matrix A_m(2, 2, 0.0);
    A_m.data = {1.0, 2.0,
                3.0, 4.0};

    math::Matrix B_m(2, 2, 0.0);
    B_m.data = {5.0, 6.0,
                7.0, 8.0};

    Tensor a(A_m, /*requires_grad=*/true);
    Tensor b(B_m, /*requires_grad=*/true);

    // c = a + b
    Tensor c = add(a, b);

    // loss = sum(c)
    Tensor loss = sum(c);

    // Backward
    backward(loss);

    // Expected:
    // c = a + b, so loss = sum(a) + sum(b)
    // d(loss)/d(c_ij) = 1
    // d(loss)/d(a_ij) = 1, d(loss)/d(b_ij) = 1

    cout << "a.grad:" << endl;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            double g = a.grad.data[i * a.cols() + j];
            cout << g << " ";
            assert(almost_equal(g, 1.0));
        }
        cout << endl;
    }

    cout << "b.grad:" << endl;
    for (std::size_t i = 0; i < b.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            double g = b.grad.data[i * b.cols() + j];
            cout << g << " ";
            assert(almost_equal(g, 1.0));
        }
        cout << endl;
    }

    cout << "All autograd tests passed!" << endl;
    return 0;
}
