#include "vocab.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <sstream>

using namespace std;

namespace data {
    Vocab::Vocab() : built(false) {};

    // Helper to split string by whitespace
    vector<string> split(const string& s) {
        vector<string> tokens;
        string token;
        istringstream tokenStream(s);
        while (getline(tokenStream, token, ' ')) {
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        return tokens;
    }

    void Vocab::build_from_text(const string& text){
        stoi.clear();
        itos.clear();
        
        unordered_set<string> uniq;
        vector<string> words = split(text);

        // Reserve helps performance
        uniq.reserve(words.size()); 

        for(const auto& w : words){
            uniq.insert(w);
        }

        // make ids deterministic, sort words
        vector<string> sorted_words(uniq.begin(), uniq.end());
        sort(sorted_words.begin(), sorted_words.end());

        itos = sorted_words;
        stoi.reserve(itos.size());

        for(size_t i = 0; i < itos.size(); i++){
            stoi[itos[i]] = static_cast<int>(i);
        }
        built = true;
    }

    vector<int> Vocab::encode(const string& s) const{
        if(!built){
            throw runtime_error("Vocab not built");
        }
        vector<int> ids;
        vector<string> words = split(s);
        ids.reserve(words.size());
        
        for(const auto& w : words){
            auto it = stoi.find(w);
            // In a real system, we'd handle <UNK>, but here we throw/skip or better:
            // If not found, we can just skip or throw. Let's throw for safety now.
            if(it == stoi.end()){
                 // For robustness, maybe skip or use a default? 
                 // But prompt implies "perfectly like how any thing built on Attention..."
                 // Usually implies a fixed vocab. Let's throw to alert mismatch.
                throw runtime_error("Vocab encode - unknown word: " + w);
            }
            ids.push_back(it->second);
        }
        return ids;
    }

    string Vocab::decode(const vector<int>& ids) const{
        if(!built){
            throw runtime_error("Vocab not built");
        }
        string s;
        
        for(size_t i = 0; i < ids.size(); ++i){
            int id = ids[i];
            if( id < 0 || static_cast<size_t>(id) >= itos.size()){
                throw runtime_error("Vocab id out of range");
            }
            s += itos[static_cast<size_t>(id)];
            if (i < ids.size() - 1) s += " ";
        }

        return s;
    }

    size_t Vocab::size() const {
        return itos.size();
    }
    bool Vocab::is_built() const {
        return built;
    }
}
