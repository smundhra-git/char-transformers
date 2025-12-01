#pragma once

#include <string>
#include <vector>
#include <unordered_map>

using namespace std;

namespace data {
    // Word-level vocab
    // Each distinct word gets an integer id
    // provide encode and decode

    class Vocab {
        public:
            Vocab();

            // builds vocab from a text corpus (splits by space)
            void build_from_text(const string& text);

            // encode a string into a sequence of integer ids
            // splits input string by space
            vector<int> encode(const string& s) const;

            // decode a sequence of ids back to a string (words joined by space)
            string decode(const vector<int>& ids) const;

            // size of the vocab
            size_t size() const;

            // check if vocab has been built
            bool is_built() const;

        private:
            unordered_map<string, int> stoi; // word->id
            vector<string> itos; // id -> word
            bool built;
        };
} // namespace data
