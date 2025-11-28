#include "self_attention.hpp"

#include <cmath>
#include <stdexcept>

#include "../math/matrix.hpp"
#include "../engine/ops.hpp"
#include "../engine/tensor.hpp"

namespace nn {
    using namespace std;
    using engine::Tensor;
    using engine::matmul;
    using engine::scale;
    using engine::softmax_row;
    using engine::transpose;
    using math::Matrix;

    //helper a casual mask matrix [T * T]
    //mask (i, j ) = 0 if j <= i else -1e9

    static Matrix build_casual_mask(size_t T) {
        Matrix m(T, T, 0.0);
        double neg = -1e9;
        for(size_t i = 0; i < T; i++){
            for(size_t j = 0; j < T; j++){
                if (j > i) {
                    m.data[i* T + j] = neg;
                } else {
                    m.data[i* T + j] = 0.0;
                }
            }
        }

        return m;
    }

    SelfAttention::SelfAttention(const SelfAttentionConfig& cfg) :
        W_q({cfg.d_model, cfg.d_model}),
        W_k({cfg.d_model, cfg.d_model}),
        W_v({cfg.d_model, cfg.d_model}),
        W_o({cfg.d_model, cfg.d_model}),
        casual(cfg.casual) 
        {
            if (cfg.d_model == 0) {
                throw runtime_error("SelfAttention : d_model must be > 0");
            }
        }

        Tensor SelfAttention::forward(Tensor& x){
            //x : [T * d_model]
            size_t T = x.rows();
            size_t d_model = x.cols();
            
            // Linear projections 
            Tensor Q = W_q.forward(x);
            Tensor K = W_k.forward(x);
            Tensor V = W_v.forward(x);

            //scores 
            Tensor K_t = transpose(K);
            Tensor scores = matmul(Q, K_t);

            double scale_factor = 1.0/sqrt(static_cast<double>(d_model));
            scores = scale(scores, scale_factor);

            //casual mask (if enabled)
            if(casual){
                Matrix mask_m = build_casual_mask(T);
                Tensor mask_t(mask_m, false);
                scores = engine::add(scores, mask_t);
            }

            //softmax ocer rows - for attention probabilities
            Tensor P = softmax_row(scores);

            //weighted sum over values Y = P * V
            Tensor Y = matmul(P, V);

            Tensor out = W_o.forward(Y);

            return out;
        }
}