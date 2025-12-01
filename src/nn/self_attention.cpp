#include "self_attention.hpp"
#include "../engine/ops.hpp"
#include <cmath>

namespace nn {
    using namespace engine;

    MultiHeadAttention::MultiHeadAttention(const SelfAttentionConfig& cfg)
        : w_q({cfg.d_model, cfg.d_model}),
          w_k({cfg.d_model, cfg.d_model}),
          w_v({cfg.d_model, cfg.d_model}),
          w_o({cfg.d_model, cfg.d_model}),
          n_head(cfg.n_head), d_model(cfg.d_model), causal(cfg.causal)
    {}

    Tensor MultiHeadAttention::forward(const Tensor& x) {
        // x: (B, T, D)
        size_t B = x.shape(0);
        size_t T = x.shape(1);
        size_t D = d_model;
        size_t H = n_head;
        size_t C = D / H; // head dim

        // Projections: (B, T, D)
        Tensor q = w_q.forward(x);
        Tensor k = w_k.forward(x);
        Tensor v = w_v.forward(x);

        // View as heads: (B, T, H, C)
        q = reshape(q, {B, T, H, C});
        k = reshape(k, {B, T, H, C});
        v = reshape(v, {B, T, H, C});

        // Transpose for attention: (B, H, T, C)
        q = permute(q, {0, 2, 1, 3});
        k = permute(k, {0, 2, 1, 3});
        v = permute(v, {0, 2, 1, 3});

        // Attention Scores: Q @ K^T
        // K is (B, H, T, C), we want K^T as (B, H, C, T)
        // transpose(tensor) flips last two dims.
        Tensor k_t = transpose(k);
        
        Tensor scores = matmul(q, k_t); // (B, H, T, T)
        scores = scale(scores, 1.0 / std::sqrt((double)C));

        // Masking
        if (causal) {
            // Create mask (T, T)
            // We need to add it to (B, H, T, T).
            // Since we lack full broadcast, let's just manually fill a tensor of same shape?
            // Or implement a `masked_fill` op. 
            // For simplicity/speed in this limited engine: 
            // I will create a "CausalMaskAdd" op in ops?
            // Or loop in C++ and add -inf.
            
            // Let's assume we have an `add_mask` op or similar. 
            // I'll create `mask_causal` in ops.cpp later?
            // I will skip mask for now to ensure compilation, but for accuracy it's needed.
            // HACK: Just rely on the model learning to ignore future? No, that breaks causality.
            // I will implement a manual causal mask loop here using data() access.
            // It's efficient enough for CPU.
            
            double neg = -1e9;
            double* ptr = scores.data().data();
            // Shape (B, H, T, T)
            size_t inner = T * T;
            size_t outer = B * H;
            
            for(size_t o=0; o<outer; ++o) {
                double* matrix = ptr + o * inner;
                for(size_t i=0; i<T; ++i) {
                    for(size_t j=i+1; j<T; ++j) { // j > i -> future
                        matrix[i * T + j] = neg;
                    }
                }
            }
        }

        Tensor attn = softmax(scores, -1); // (B, H, T, T)
        
        // Output: Attn @ V
        // (B, H, T, T) @ (B, H, T, C) -> (B, H, T, C)
        // Wait, V is (B, H, T, C).
        // Matmul last 2 dims: (T, T) @ (T, C) -> (T, C). Correct.
        Tensor y = matmul(attn, v);

        // Reassemble: (B, T, H, C)
        y = permute(y, {0, 2, 1, 3}); 
        
        // Flatten: (B, T, D)
        y = reshape(y, {B, T, D});

        return w_o.forward(y);
    }
}
