#pragma once

#include <vector>

using namespace std;

namespace data {

    //a sample batch of int tokens ids
    //x and y are [batch size * block size] in row-major
    struct Batch {
        size_t batch_size;
        size_t block_size;

        //flattened 2D arrays 
        vector<int> x;
        vector<int> y;
    };

    //char level dataset over tokenized corpus

    //tokens - full coropus as int ids
    //block_size - context lebtgh

    //we will start a simple "sequential chunks" sampler;
    //each call to next_batch() moves a cursor forward

    class CharDataset {
        public :
            CharDataset(vector<int>& tokens, size_t block_size);

            //get the next batch of (x,y) sequence; wrap around at end
            Batch next_batch(size_t batch_size);

            size_t get_block_size() const;

            size_t num_tokens() const;

        private:
            vector<int> tokens;
            size_t block_size;
            size_t cursor; //position to start next batch

    };
}