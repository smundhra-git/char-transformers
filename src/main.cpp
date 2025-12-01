#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "nn/char_transformer.hpp" // Defines nn::Transformer
#include "optim/adam.hpp"
#include "data/vocab.hpp"
#include "data/dataset.hpp"

using namespace std;
using namespace engine;
using namespace nn;

// Helper to get all parameters
void get_params(vector<Tensor>& params, Transformer& model) {
    params.push_back(model.tok_emb.W);
    params.push_back(model.ln_f_gamma);
    params.push_back(model.ln_f_beta);
    params.push_back(model.lm_head.W);
    params.push_back(model.lm_head.b);
    
    for(auto& b : model.blocks) {
        params.push_back(b.sa.w_q.W);
        params.push_back(b.sa.w_k.W);
        params.push_back(b.sa.w_v.W);
        params.push_back(b.sa.w_o.W);
        params.push_back(b.ff.c_fc.W);
        params.push_back(b.ff.c_fc.b);
        params.push_back(b.ff.c_proj.W);
        params.push_back(b.ff.c_proj.b);
        params.push_back(b.ln1_gamma);
        params.push_back(b.ln1_beta);
        params.push_back(b.ln2_gamma);
        params.push_back(b.ln2_beta);
    }
}

// Multinomial sampling with temperature
int sample(const Tensor& logits, double temperature = 1.0) {
    // logits: (1, T, V) -> look at last time step
    size_t V = logits.shape(2);
    size_t T = logits.shape(1);
    const double* ptr = logits.data().data() + (T - 1) * V;

    // 1. Apply temperature and exp (Softmax)
    std::vector<double> probs(V);
    double max_val = -1e9;
    for(size_t i=0; i<V; ++i) {
        if(ptr[i] > max_val) max_val = ptr[i];
    }

    double sum = 0.0;
    for(size_t i=0; i<V; ++i) {
        probs[i] = std::exp((ptr[i] - max_val) / temperature);
        sum += probs[i];
    }

    // 2. Sample
    static std::mt19937 gen(1337); 
    std::uniform_real_distribution<> dist(0.0, sum);
    double r = dist(gen);
    
    double cum = 0.0;
    for(size_t i=0; i<V; ++i) {
        cum += probs[i];
        if(r <= cum) return (int)i;
    }
    return (int)V - 1;
}

int main() {
    // @zsh (997-1020)
    cout << "=== WordTransformer (N-Dim Engine) ===" << endl;
    cout << "Training a Transformer from scratch in C++ with Word Embeddings and Sinusoidal PE..." << endl;

    // 1. Data
    string source_text = "Hello world! This is a test of the N-dimensional tensor engine written from scratch in C++. It supports batching, multi-head attention, and layer norm. It is based on Attention Is All You Need.";
    string text = "";
    // Repeat it enough times to form a dataset
    for(int i=0; i<100; ++i) text += source_text + " "; 

    data::Vocab vocab;
    vocab.build_from_text(text);
    vector<int> tokens = vocab.encode(text);

    cout << "Vocab Size: " << vocab.size() << endl;
    cout << "Total Text Length (Words): " << tokens.size() << endl;

    // 2. Config
    size_t block_size = 16; 
    TransformerConfig cfg;
    cfg.vocab_size = vocab.size();
    cfg.d_model = 64;      
    cfg.n_head = 4;        
    cfg.n_layer = 4;       
    cfg.block_size = block_size;
    cfg.d_ff = 4 * cfg.d_model;

    Transformer model(cfg);
    
    vector<Tensor> params;
    get_params(params, model);
    cout << "Model Params: " << params.size() << endl;

    // 3. Train
    // Adam usually prefers lower LR than SGD (e.g., 1e-3)
    double lr = 0.0005; 
    int steps = 2000; // Increased steps for better convergence
    int batch_size = 4;

    data::CharDataset dataset(tokens, block_size);
    
    // Use Adam Optimizer
    optim::Adam optimizer(params, lr);

    auto start_time = std::chrono::high_resolution_clock::now();

    cout << "Starting training (Steps: " << steps << ", Batch Size: " << batch_size << ")..." << endl;

    for(int step=0; step<steps; ++step) {
        data::Batch b = dataset.next_batch(batch_size);
        
        // Forward
        Tensor logits = model.forward(b.x, batch_size);
        
        // Loss
        Tensor target(vector<double>(b.y.begin(), b.y.end()), { (size_t)batch_size, block_size }, false);
        Tensor loss = cross_entropy(logits, target);

        // Zero Grad
        optimizer.zero_grad();

        // Backward
        backward(loss);

        // Step
        optimizer.step();

        // @zsh (1004-1013) Logging Loss
        if(step % 100 == 0) {
            cout << "Step " << step << " | Loss: " << loss.data()[0] << endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    cout << "Training completed in " << sec << "s." << endl;

    // 4. Generate with Sampling
    cout << "\n=== Generating Text ===" << endl;
    // @zsh (1017) Prompt
    string prompt = "Hello world! This is";
    cout << "Prompt: '" << prompt << "'" << endl;
    vector<int> ctx = vocab.encode(prompt);
    
    for(int t : ctx) cout << vocab.decode({t}) << " ";

    // Generate more words
    for(size_t i=0; i<100; ++i) {
        vector<int> input = ctx;
        if(input.size() > block_size) 
            input.erase(input.begin(), input.end() - block_size);
        
        Tensor logits = model.forward(input, 1);
        // Lower temp for more accurate reproduction of learned text
        int next_token = sample(logits, 0.1); 
        
        ctx.push_back(next_token);
        cout << vocab.decode({next_token}) << " " << std::flush;
    }
    cout << endl;

    return 0;
}
