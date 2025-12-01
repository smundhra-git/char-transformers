#include "self_attention.hpp"
#include "../engine/ops.hpp"
#include <cmath>
#include <iostream>

namespace nn {
    using namespace engine;

    MultiHeadAttention::MultiHeadAttention(const SelfAttentionConfig& cfg)
        : w_q({cfg.d_model, cfg.d_model}),
          w_k({cfg.d_model, cfg.d_model}),
          w_v({cfg.d_model, cfg.d_model}),
          w_o({cfg.d_model, cfg.d_model}),
          n_head(cfg.n_head), d_model(cfg.d_model), causal(cfg.causal)
    {}

    Tensor MultiHeadAttention::forward(const Tensor& x, KVCache* cache) {
        // x: (B, T, D)
        size_t B = x.shape(0);
        size_t T = x.shape(1);
        size_t D = d_model;
        size_t H = n_head;
        size_t C = D / H; 

        // 1. Compute Q, K, V for current input
        Tensor q = w_q.forward(x);
        Tensor k = w_k.forward(x);
        Tensor v = w_v.forward(x);

        // Reshape to (B, T, H, C)
        q = reshape(q, {B, T, H, C});
        k = reshape(k, {B, T, H, C});
        v = reshape(v, {B, T, H, C});

        // Permute to (B, H, T, C)
        q = permute(q, {0, 2, 1, 3});
        k = permute(k, {0, 2, 1, 3});
        v = permute(v, {0, 2, 1, 3});
        
        // Ensure K, V are contiguous for caching/matmul
        q = q.contiguous();
        k = k.contiguous();
        v = v.contiguous();
        
        // KV Caching
        if (cache) {
            // If cache is empty, init it
            // BEFORE:
            // if (cache->k.numel() == 0) {
            // AFTER (use current_len, which is 0 for a fresh cache):
            if (cache->current_len == 0 || cache->k.shape().empty()) {
                cache->k = k; 
                cache->v = v;
                cache->current_len = T;
            } else {
                // Manual Concat
                Shape old_shape = cache->k.shape();
                size_t old_T = old_shape[2];
                size_t new_T = old_T + T;
                
                Shape new_shape = old_shape;
                new_shape[2] = new_T;
                
                Tensor next_k(new_shape, false);
                Tensor next_v(new_shape, false);
                
                size_t chunk_old = old_T * C;
                size_t chunk_new = T * C;
                size_t chunk_total = new_T * C;
                
                size_t outer = B * H;
                
                double* dst_k = next_k.data().data() + next_k.offset();
                double* dst_v = next_v.data().data() + next_v.offset();
                const double* src_old_k = cache->k.data().data() + cache->k.offset();
                const double* src_old_v = cache->v.data().data() + cache->v.offset();
                const double* src_new_k = k.data().data() + k.offset();
                const double* src_new_v = v.data().data() + v.offset();
                
                for(size_t i=0; i<outer; ++i) {
                     std::copy(src_old_k + i * chunk_old, src_old_k + (i+1) * chunk_old, dst_k + i * chunk_total);
                     std::copy(src_old_v + i * chunk_old, src_old_v + (i+1) * chunk_old, dst_v + i * chunk_total);
                     
                     std::copy(src_new_k + i * chunk_new, src_new_k + (i+1) * chunk_new, dst_k + i * chunk_total + chunk_old);
                     std::copy(src_new_v + i * chunk_new, src_new_v + (i+1) * chunk_new, dst_v + i * chunk_total + chunk_old);
                }
                
                cache->k = next_k;
                cache->v = next_v;
                cache->current_len = new_T;
            }
            
            // Use cached K, V for attention
            k = cache->k;
            v = cache->v;
        }

        // Attention: q @ k^T
        // k is (B, H, T_total, C). transpose -> (B, H, C, T_total) (contiguous copy in ops)
        Tensor k_t = transpose(k);
        Tensor scores = matmul(q, k_t);
        scores = scale(scores, 1.0 / std::sqrt((double)C));

        // Masking
        if (causal) {
            if (!cache && T > 1) {
                 double neg = -1e9;
                 double* ptr = scores.data().data() + scores.offset();
                 // scores is (B, H, T, T)
                 size_t inner = T * T; 
                 size_t outer = B * H;
                 
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for(size_t o=0; o<outer; ++o) {
                     double* matrix = ptr + o * inner;
                     for(size_t i=0; i<T; ++i) {
                         for(size_t j=i+1; j<T; ++j) {
                             matrix[i * T + j] = neg;
                         }
                     }
                 }
            }
        }

        Tensor attn = softmax(scores, -1); 
        Tensor y = matmul(attn, v); 

        y = permute(y, {0, 2, 1, 3}); // (B, T, H, C)
        y = reshape(y, {B, T, D});    // (B, T, D)

        return w_o.forward(y);
    }
}
