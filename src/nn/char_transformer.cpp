#include "char_transformer.hpp"
#include "../engine/ops.hpp"

using namespace std;

namespace nn {
    using namespace engine;

    Transformer::Transformer(const TransformerConfig& cfg)
        : tok_emb({cfg.vocab_size, cfg.d_model}),
          pos_enc(cfg.d_model, 5000), // Fix: Allow larger max_len for generation
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

        Tensor tok = tok_emb.forward(idx, Batch, T);
        Tensor pos = pos_enc.forward(T);
        
        Tensor pos_b({Batch, T, tok.shape(2)}, false);
        const double* p_src = pos.data().data() + pos.offset();
        double* p_dst = pos_b.data().data() + pos_b.offset();
        size_t vol = T * tok.shape(2);
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(size_t b=0; b<Batch; ++b) {
            std::copy(p_src, p_src + vol, p_dst + b * vol);
        }
        
        Tensor x = add(tok, pos_b);

        for(auto& block : blocks) {
            x = block.forward(x, nullptr); 
        }

        x = layer_norm(x, ln_f_gamma, ln_f_beta);
        return lm_head.forward(x);
    }

    Tensor Transformer::forward_generate(int token, std::vector<KVCache>& caches) {
        // Debug
        // std::cout << "Gen token: " << token << std::endl;
        Tensor tok = tok_emb.forward({token}, 1, 1);
        
        size_t current_pos = caches[0].current_len;
        // std::cout << "Pos: " << current_pos << std::endl;
        
        // Fix: Ensure we don't exceed max_len of pos_enc (5000).
        if (current_pos >= 5000) throw std::runtime_error("Exceeded max generation length");

        Tensor full_pos = pos_enc.forward(current_pos + 1); 
        
        size_t D = tok.shape(2);
        Tensor pos_i({1, 1, D}, false);

        
        // Fix: Add offset to full_pos and pos_i
        const double* src = full_pos.data().data() + full_pos.offset() + current_pos * D;
        double* dst = pos_i.data().data() + pos_i.offset();
        std::copy(src, src + D, dst);
        
        Tensor x = add(tok, pos_i);

        
        for(size_t i=0; i<blocks.size(); ++i) {
            x = blocks[i].forward(x, &caches[i]);
        }
        x = layer_norm(x, ln_f_gamma, ln_f_beta);
        return lm_head.forward(x);
    }
}
