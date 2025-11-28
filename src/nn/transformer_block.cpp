#include "transformer_block.hpp"

#include "../engine/ops.hpp"
#include <stdexcept>

using namespace std;

namespace nn {

    using engine::Tensor;
    using engine::add;
    using engine::relu;

    TransformerBlock::TransformerBlock(const TransformerBlockConfig& cfg):
    sa({cfg.d_model, cfg.casual}),
    ff1({cfg.d_model, cfg.d_ff}),
    ff2({cfg.d_ff, cfg.d_model}){
        if(cfg.d_model == 0 || cfg.d_ff == 0){
            throw runtime_error("Transformer Block - d_model and d_ff must be >0");
        }
    }

    Tensor TransformerBlock::forward(Tensor& x){
        //x : [T * d_model]
        //self attention sublayer
        Tensor h = sa.forward(x);

        //residual connection
        Tensor x1 = add(x, h);

        //feedforward sublayer 
        Tensor f1 = ff1.forward(x1);
        Tensor f1_act = relu(f1);
        Tensor f2 = ff2.forward(f1_act);

        //residual connection 

        Tensor y = add(x1, f2);

        return y;
    }
}