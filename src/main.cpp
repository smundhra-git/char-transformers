#include <iostream>
#include "data/vocab.hpp"
#include "nn/embedding.hpp"
#include "engine/ops.hpp"
#include "engine/tensor.hpp"

using namespace std;
using namespace data;
using namespace nn;
using namespace engine;

int main() {
    cout << "=== Embedding test ===" << endl;

    string corpus = "abca";
    Vocab vocab;
    vocab.build_from_text(corpus);

    vector<int> ids = vocab.encode("ab");
    // vocab.size() maybe 2 or 3 depending on chars

    EmbeddingConfig cfg{ vocab.size(), 4 }; // d_model = 4
    Embedding emb(cfg);

    Tensor E = emb.forward(ids);  // [2 x 4]

    cout << "E rows: " << E.rows() << ", cols: " << E.cols() << endl;

    // Loss: sum(E) just to trigger grads
    Tensor loss = sum(E);
    backward(loss);

    cout << "W.grad (some rows should be non-zero):" << endl;
    for (size_t i = 0; i < emb.W.rows(); ++i) {
        for (size_t j = 0; j < emb.W.cols(); ++j) {
            cout << emb.W.grad.data[i * emb.W.cols() + j] << " ";
        }
        cout << endl;
    }

    return 0;
}
