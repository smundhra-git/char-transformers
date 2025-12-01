#include "embedding.hpp"
#include "../engine/node.hpp"
#include <stdexcept>

namespace nn {
    using namespace engine;
    using namespace std;

    struct EmbeddingNode : public Node {
        vector<int> ids;
        EmbeddingNode(const vector<int>& i) : ids(i) {}

        void backward(const vector<double>& grad_out) override {
            Tensor& W = inputs[0];
            if (!W.require_grad()) return;

            // grad_out matches output shape (B, T, D)
            // ids match (B, T)
            size_t D = W.shape(1);
            size_t N = ids.size();

            // Sparse update
            double* w_grad = W.grad().data();
            const double* g = grad_out.data();

            for (size_t i = 0; i < N; ++i) {
                int idx = ids[i];
                // Accumulate gradient
                double* row = w_grad + idx * D;
                const double* g_row = g + i * D;
                for (size_t j = 0; j < D; ++j) {
                    row[j] += g_row[j];
                }
            }
        }
    };

    Embedding::Embedding(const EmbeddingConfig& cfg) {
        W = Tensor::randn({cfg.vocab_size, cfg.d_model}, 0.0, 0.02, true);
    }

    Tensor Embedding::forward(const std::vector<int>& token_ids, size_t Batch, size_t T) {
        size_t N = token_ids.size();
        if (N != Batch * T) throw runtime_error("Embedding forward: IDs size mismatch");

        size_t D = W.shape(1);
        Tensor out({Batch, T, D}, true);

        const double* w_data = W.data().data();
        double* out_data = out.data().data();

        for (size_t i = 0; i < N; ++i) {
            int idx = token_ids[i];
            if (idx < 0 || idx >= (int)W.shape(0)) throw runtime_error("Token ID out of bounds");
            
            const double* src = w_data + idx * D;
            double* dst = out_data + i * D;
            std::copy(src, src + D, dst);
        }

        if (out.require_grad()) {
            auto node = make_shared<EmbeddingNode>(token_ids);
            node->inputs = {W};
            out.grad_fn() = node;
        }
        return out;
    }
}
