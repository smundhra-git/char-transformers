#include <iostream>
#include <cmath>
#include <vector>

#include "math/matrix.hpp"
#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "nn/linear.hpp"
#include "optim/sgd.hpp"

using namespace std;
using namespace math;
using namespace engine;
using namespace nn;
using namespace optim;

int main() {
    cout << "=== 2-layer MLP on XOR (1-sample SGD) ===" << endl;

    // Model: 2 -> 4 -> 1
    LinearConfig cfg1{2, 4};
    LinearConfig cfg2{4, 1};

    Linear layer1(cfg1);
    Linear layer2(cfg2);

    // XOR dataset: inputs and targets
    struct Sample { double x1, x2, y; };
    vector<Sample> data = {
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0}
    };

    double lr = 0.1;
    int epochs = 5000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (const auto& s : data) {
            // Build input x: [1 x 2]
            Matrix X_m(1, 2, 0.0);
            X_m.data = { s.x1, s.x2 };
            Tensor x(X_m, /*require_grad=*/false);

            // Target y_true: [1 x 1]
            Matrix Y_m(1, 1, 0.0);
            Y_m.data = { s.y };
            Tensor y_true(Y_m, /*require_grad=*/false);

            // Forward: x -> layer1 -> relu -> layer2 -> y_pred
            Tensor h1     = layer1.forward(x);      // [1 x 4]
            Tensor a1     = relu(h1);               // [1 x 4]
            Tensor y_pred = layer2.forward(a1);     // [1 x 1]

            // Loss
            Tensor diff = sub(y_pred, y_true);
            Tensor sq = hadamard(diff, diff);
            Tensor loss = sq;
            epoch_loss += loss.data.data[0];

            // Backprop
            backward(loss);

            // SGD step on parameters
            sgd_step({ &layer1.W, &layer1.b,
                       &layer2.W, &layer2.b }, lr);
        }

        epoch_loss /= static_cast<double>(data.size());

        if (epoch % 500 == 0) {
            cout << "Epoch " << epoch
                 << "  loss = " << epoch_loss << endl;
        }
    }

    // Final predictions
    cout << "\nFinal predictions:" << endl;
    for (const auto& s : data) {
        Matrix X_m(1, 2, 0.0);
        X_m.data = { s.x1, s.x2 };
        Tensor x(X_m, /*require_grad=*/false);

        Tensor h1     = layer1.forward(x);
        Tensor a1     = relu(h1);
        Tensor y_pred = layer2.forward(a1);

        double val = y_pred.data.data[0];
        cout << "(" << s.x1 << ", " << s.x2 << ") -> "
             << val << " (target " << s.y << ")" << endl;
    }

    return 0;
}
