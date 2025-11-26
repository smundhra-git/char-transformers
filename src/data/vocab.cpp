#include "vocab.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

using namespace std;

namespace data{
    Vocab::Vocab() : built(false) {};

    void Vocab::build_from_text(const string& text){
        stoi.clear();
        itos.clear();
        
        unordered_set<char> uniq;

        uniq.reserve(text.size()); //at worse all unique

        for(char c : text){
            uniq.insert(c);
        }

        //make ids deterministic, sort chars
        vector<char> chars(uniq.begin(), uniq.end());
        sort(chars.begin(), chars.end());

        itos = chars;
        stoi.reserve(itos.size());

        for(size_t i = 0; i< itos.size(); i++){
            stoi[itos[i]] = static_cast<int>(i);
        }
        built = true;
    }

    vector<int> Vocab::encode(const string& s) const{
        if(!built){
            throw runtime_error("Vocab not built");
        }
        vector<int> ids;
        ids.reserve(s.size());
        for(char c : s){
            auto it = stoi.find(c);
            if(it == stoi.end()){
                throw runtime_error("Vocab encode - unknown char");
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
        s.reserve(ids.size());

        for(int id : ids){
            if( id < 0 || static_cast<size_t>(id) >= itos.size()){
                throw runtime_error("Vocab id out of range");
            }
            s.push_back(itos[static_cast<size_t>(id)]);
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