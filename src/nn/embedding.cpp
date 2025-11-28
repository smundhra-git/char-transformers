#include "embedding.hpp"

#include "../math/matrix.hpp"
#include "../engine/node.hpp"
#include <random>
#include <stdexcept>

namespace nn {
    using namespace std;
    using engine::Tensor;
    using math::Matrix;

    //autograd model for embeddings lookup
    //out[p, :] = W[token_ids[p], :]
    struct EmbeddingNode : public engine::Node {
        vector<int> token_ids; //size = N

        explicit EmbeddingNode(const vector<int>& ids)
            : token_ids(ids) {}
        

        void backward(const math::Matrix& grad_out) override {
            //grad_out = [N* d_model]

            Tensor& W = inputs[0];

            if(!W.require_grad()) return;

            size_t N = grad_out.rows;
            size_t d_model = grad_out.cols;

            if (N != token_ids.size()){
                throw runtime_error("Size mismatch in embedding - backward");
            }

            //Ensure W.grad has correct shape : [vocab_size * d_model]
            W.grad().resize(W.rows(), W.cols(), 0.0);

            for(size_t p = 0; p< N; p++){
                int token_id = token_ids[p];
                if(token_id < 0 || static_cast<size_t>(token_id) >= W.rows()){
                    throw runtime_error("Embed - backward token id out of range");
                }

                size_t row_w = static_cast<size_t>(token_id);

                for(size_t j = 0; j < d_model; j++){
                    double g = grad_out.data[p * d_model + j];
                    W.grad().data[row_w * d_model + j] += g;
                }
            }

        }
    };

    Embedding::Embedding(const EmbeddingConfig& cfg) : W(){
        if(cfg.vocab_size == 0 || cfg.d_model == 0){
            throw runtime_error("Embedding - vocab size and d_model must be > 0");
        }


        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> distrib(-0.1, 0.1);

        Matrix w_matrix(cfg.vocab_size, cfg.d_model, 0.0);
        for(size_t i = 0; i<w_matrix.size(); i++){
            w_matrix.data[i] = distrib(gen);
        }

        W = Tensor(w_matrix, true);

    }

    Tensor Embedding::forward(const std::vector<int>& token_ids) {
        if (token_ids.empty()) {
            throw std::runtime_error("Embedding::forward: token_ids empty");
        }

        std::size_t N       = token_ids.size();
        std::size_t d_model = W.cols();

        // Forward: build [N x d_model] matrix
        Matrix out_data(N, d_model, 0.0);

        for (std::size_t p = 0; p < N; ++p) {
            int token_id = token_ids[p];
            if (token_id < 0 || static_cast<std::size_t>(token_id) >= W.rows()) {
                throw std::runtime_error("Embedding::forward: token id out of range");
            }

            std::size_t row_w = static_cast<std::size_t>(token_id);
            for (std::size_t j = 0; j < d_model; ++j) {
                out_data.data[p * d_model + j] =
                    W.data().data[row_w * d_model + j];
            }
        }

        Tensor out(out_data, true);

        // Register autograd node
        auto node = std::make_shared<EmbeddingNode>(token_ids);
        node->inputs = { W};  // only param W gets gradients
        out.p->grad_fn = node;

        return out;
    }

}