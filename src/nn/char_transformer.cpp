#include "char_transformer.hpp"
#include "../engine/ops.hpp"

namespace nn {
    using namespace engine;

    Transformer::Transformer(const TransformerConfig& cfg)
        : tok_emb({cfg.vocab_size, cfg.d_model}),
          pos_enc(cfg.d_model, cfg.block_size), // Use fixed Sinusoidal PE
          lm_head({cfg.d_model, cfg.vocab_size}),
          block_size(cfg.block_size)
    {
        for(size_t i=0; i<cfg.n_layer; ++i) {
            blocks.emplace_back(SelfAttentionConfig{cfg.d_model, cfg.n_head, cfg.block_size, true}, cfg.d_ff);
        }
        
        ln_f_gamma = Tensor::constant({cfg.d_model}, 1.0, true);
        ln_f_beta = Tensor::constant({cfg.d_model}, 0.0, true);
    }

    Tensor Transformer::forward(const std::vector<int>& idx, size_t Batch) {
        size_t N = idx.size();
        if (N % Batch != 0) throw std::runtime_error("Batch size mismatch");
        size_t T = N / Batch;
        if (T > block_size) throw std::runtime_error("Seq len > block size");

        // 1. Token embeddings (B, T, D)
        Tensor tok = tok_emb.forward(idx, Batch, T);

        // 2. Positional Encodings (1, T, D) using Sinusoidal formulation
        //    Fixed, not learned.
        Tensor pos = pos_enc.forward(T);
        
        // Broadcast pos to (B, T, D) manually and add to token embeddings
        Tensor pos_b({Batch, T, tok.shape(2)}, false);
        const double* p_src = pos.data().data();
        double* p_dst = pos_b.data().data();
        size_t vol = T * tok.shape(2);
        for(size_t b=0; b<Batch; ++b) {
            std::copy(p_src, p_src + vol, p_dst + b * vol);
        }
        
        Tensor x = add(tok, pos_b);

        // 3. Blocks (Multi-Head Attention + Feed Forward)
        for(auto& block : blocks) {
            x = block.forward(x);
        }

        // 4. Final Layer Norm
        x = layer_norm(x, ln_f_gamma, ln_f_beta);

        // 5. Language Model Head (Projection to Vocab)
        return lm_head.forward(x);
    }
}
