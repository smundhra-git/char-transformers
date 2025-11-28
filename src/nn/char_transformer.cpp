#include "char_transformer.hpp"

#include "../engine/ops.hpp"

namespace nn {

    using engine::Tensor;
    using namespace std;

    CharTransformer::CharTransformer(const CharTransformerConfig& cfg):
    emb({cfg.vocab_size, cfg.d_model}),
    block({cfg.d_model, cfg.d_ff, cfg.casual}),
    lm_head({cfg.d_model, cfg.vocab_size}) {}

    Tensor CharTransformer::forward(const vector<int>& ids){
        //ids.size() = T
        Tensor x = emb.forward(ids);
        Tensor h = block.forward(x);
        Tensor logits = lm_head.forward(h);

        return logits;
    }
}