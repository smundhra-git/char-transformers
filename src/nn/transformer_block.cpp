#include "transformer_block.hpp"
#include "../engine/ops.hpp"

#include <iostream> //Debug
using namespace std;

namespace nn {
    using namespace engine;

    MLP::MLP(size_t d_model, size_t d_ff) 
        : c_fc({d_model, d_ff}), c_proj({d_ff, d_model}) 
    {}

    Tensor MLP::forward(const Tensor& x) {
        return c_proj.forward(gelu(c_fc.forward(x)));
    }

    Block::Block(const SelfAttentionConfig& cfg, size_t d_ff) 
        : sa(cfg), ff(cfg.d_model, d_ff)
    {
        ln1_gamma = Tensor::constant({cfg.d_model}, 1.0, true);
        ln1_beta = Tensor::constant({cfg.d_model}, 0.0, true);
        ln2_gamma = Tensor::constant({cfg.d_model}, 1.0, true);
        ln2_beta = Tensor::constant({cfg.d_model}, 0.0, true);
    }

    Tensor Block::forward(const Tensor& x, KVCache* cache) {
        Tensor ln1 = layer_norm(x, ln1_gamma, ln1_beta);
        Tensor attn = sa.forward(ln1, cache);
        Tensor x2 = add(x, attn);
        
        Tensor ln2 = layer_norm(x2, ln2_gamma, ln2_beta);
        Tensor mlp = ff.forward(ln2);
        return add(x2, mlp);
    }
}
