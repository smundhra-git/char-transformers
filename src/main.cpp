#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>

#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "nn/char_transformer.hpp" 
#include "optim/adam.hpp"
#include "data/vocab.hpp"
#include "data/dataset.hpp"

using namespace std;
using namespace engine;
using namespace nn;

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

int sample(const Tensor& logits, double temperature = 1.0) {
    // logits: (1, 1, V) for generation
    size_t V = logits.shape(2);
    const double* ptr = logits.data().data(); // Only 1 row

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

std::string read_file(const std::string& path) {
    std::ifstream t(path);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

int main() {
    cout << "=== Optimized WordTransformer (OpenMP + KV Cache + ZeroCopy) ===" << endl;

    // 1. Data
    string raw_path = "data/raw.txt";
    cout << "Reading data from " << raw_path << "..." << endl;
    string text = read_file(raw_path);
    if (text.empty()) {
        cerr << "Error: data/raw.txt is empty or not found." << endl;
        return 1;
    }

    data::Vocab vocab;
    vocab.build_from_text(text);
    vector<int> tokens = vocab.encode(text);

    cout << "Vocab Size: " << vocab.size() << endl;
    cout << "Total Text Length (Words): " << tokens.size() << endl;

    // 2. Config
    size_t block_size = 32; 
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
    double lr = 0.003; // Increased from 0.001
    int steps = 3000;  // Increased from 2000
    int batch_size = 32; // Increased from 8

    data::CharDataset dataset(tokens, block_size);
    optim::Adam optimizer(params, lr);

    auto start_time = std::chrono::high_resolution_clock::now();
    cout << "Starting training (Steps: " << steps << ", Batch Size: " << batch_size << ")..." << endl;

    for(int step=0; step<steps; ++step) {
        data::Batch b = dataset.next_batch(batch_size);
        
        Tensor logits = model.forward(b.x, batch_size);
        Tensor target(vector<double>(b.y.begin(), b.y.end()), { (size_t)batch_size, block_size }, false);
        Tensor loss = cross_entropy(logits, target);

        optimizer.zero_grad();
        backward(loss);
        optimizer.step();

        if(step % 50 == 0) {
            cout << "Step " << step << " | Loss: " << loss.data()[0] << endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    cout << "Training completed in " << sec << "s." << endl;

    // 4. Interactive Generation Loop
    while (true) {
        cout << "\n\n=== Interactive Generator ===" << endl;
        cout << "Enter prompt (or 'q' to quit): ";
        string prompt;
        getline(cin, prompt);
        
        if (prompt == "q") break;
        if (prompt.empty()) continue;

        try {
            vector<int> ctx = vocab.encode(prompt);
            cout << "Generating..." << endl;
            cout << prompt << " " << flush;

            // Initialize KV Caches
            vector<KVCache> caches(cfg.n_layer);

            // Warmup Cache
            for(size_t i=0; i<ctx.size()-1; ++i) {
                // cout << "[DEBUG] Warmup token " << i << endl;
                model.forward_generate(ctx[i], caches);
            }
            
            int next_token = ctx.back(); 
            // Generate 50 words
            for(size_t i=0; i<50; ++i) {
                // cout << "[DEBUG] Gen step " << i << " with token " << next_token << endl;
                Tensor logits = model.forward_generate(next_token, caches);
                next_token = sample(logits, 0.6); 
                cout << vocab.decode({next_token}) << " " << flush;
            }
        } catch (const exception& e) {
            cout << "Error: " << e.what() << " (Likely unknown word in prompt)" << endl;
        }
    }

    return 0;
}
