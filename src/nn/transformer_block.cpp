#include "transformer_block.hpp"
#include "../engine/ops.hpp"

namespace nn {
    using namespace engine;

    MLP::MLP(size_t d_model, size_t d_ff) 
        : c_fc({d_model, d_ff}), c_proj({d_ff, d_model}) {}

    Tensor MLP::forward(const Tensor& x) {
        // x: (B, T, D) -> (B, T, d_ff) -> GELU -> (B, T, D)
        Tensor h = c_fc.forward(x);
        h = gelu(h); // using gelu
        return c_proj.forward(h);
    }

    Block::Block(const SelfAttentionConfig& cfg, size_t d_ff)
        : sa(cfg), ff(cfg.d_model, d_ff)
    {
        // Init LayerNorm params
        // Shape, Value, RequireGrad
        ln1_gamma = Tensor::constant({cfg.d_model}, 1.0, true);
        ln1_beta = Tensor::constant({cfg.d_model}, 0.0, true);
        ln2_gamma = Tensor::constant({cfg.d_model}, 1.0, true);
        ln2_beta = Tensor::constant({cfg.d_model}, 0.0, true);
    }

    Tensor Block::forward(const Tensor& x) {
        // Pre-norm formulation
        // x = x + sa(ln1(x))
        // x = x + ff(ln2(x))
        
        Tensor n1 = layer_norm(x, ln1_gamma, ln1_beta);
        Tensor attn_out = sa.forward(n1);
        Tensor x2 = add(x, attn_out);
        
        Tensor n2 = layer_norm(x2, ln2_gamma, ln2_beta);
        Tensor ff_out = ff.forward(n2);
        return add(x2, ff_out);
    }
}
