#pragma once
#include "../engine/tensor.hpp"
#include <cmath>
#include <vector>

namespace nn {
    // Sinusoidal Positional Encoding as per "Attention Is All You Need"
    // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    class PositionalEncoding {
    public:
        size_t d_model;
        size_t max_len;
        engine::Tensor pe;

        PositionalEncoding(size_t d_model, size_t max_len = 5000) 
            : d_model(d_model), max_len(max_len), pe({max_len, d_model}, false) {
            
            // Precompute the positional encoding matrix
            // Shape: (max_len, d_model)
            std::vector<double> data(max_len * d_model);
            
            for (size_t pos = 0; pos < max_len; ++pos) {
                for (size_t i = 0; i < d_model; ++i) {
                    // 2i or 2i+1 logic
                    // Standard impl: i goes from 0 to d_model-1
                    // Exponent is (2 * (i//2)) / d_model
                    
                    double div_term = std::exp((i / 2 * 2) * (-std::log(10000.0) / d_model));
                    
                    if (i % 2 == 0) {
                        data[pos * d_model + i] = std::sin(pos * div_term);
                    } else {
                        data[pos * d_model + i] = std::cos(pos * div_term);
                    }
                }
            }
            
            // Manually set data to tensor (avoiding grad since it's fixed)
            // We need to use a method to set data if Tensor allows, or construct with data.
            // Tensor constructor with vector<double> exists.
            pe = engine::Tensor(data, {max_len, d_model}, false); // false = no grad
        }

        // Returns (1, T, D) tensor of positional encodings for T steps
        engine::Tensor forward(size_t T) {
            if (T > max_len) throw std::runtime_error("Sequence length exceeds max_len in PositionalEncoding");
            
            // Slice the first T rows
            // We can't easily slice with current Tensor engine (maybe?), 
            // so let's just copy the data for now or assume ops supports slicing.
            // Looking at ops.hpp, we might not have advanced slicing.
            // Let's create a new Tensor by copying relevant part.
            
            std::vector<double> subset_data(T * d_model);
            const double* src = pe.data().data();
            std::copy(src, src + T * d_model, subset_data.begin());
            
            return engine::Tensor(subset_data, {1, T, d_model}, false);
        }
    };
}

