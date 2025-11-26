#pragma once

#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

namespace data {
    //char level vocab
    //eahc distict character gets an intger id
    //provide encode and decode

    class Vocab {
        public:
            Vocab();

            //builds vocab from a text cirpus
            //scans the corpus and collects unique chars, assigns etc
            void build_from_text(const string& text);

            //encode a string into a sequence of interger id
            //require that the build has been called
            vector<int> encode(const string& s) const;

            //decode a squence of ids back to a string
            string decode(const vector<int>& ids) const;

            //size of the vocab
            size_t size() const;

            //check if vocab has been built
            bool is_built() const;

        private:
            unordered_map<char, int> stoi; //char->id
            vector<char> itos; //id -> char
            bool built;
        };
} //namespace data