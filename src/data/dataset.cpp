#include "dataset.hpp"

#include <stdexcept>

namespace data {
    CharDataset::CharDataset(vector<int>& tokens_, size_t block_size_)
    : tokens(tokens_), block_size(block_size_), cursor(0){
        if(tokens.size() < block_size + 1) {
            throw runtime_error("not enough tokens for given block size");
        }
    }

    Batch CharDataset::next_batch(size_t batch_size){
        if(tokens.size() < block_size + 1){
            throw runtime_error("next batch corpus is to small");
        }
        Batch batch;
        batch.batch_size = batch_size;
        batch.block_size = block_size;
        batch.x.assign(batch_size*block_size, 0);
        batch.y.assign(batch_size*block_size, 0);

        //we only choose start positions upto (token size - blocksize - 1)
        //becasue we access start+blocksize for target

        size_t max_start = tokens.size() - block_size - 1;
        if(max_start == 0) max_start = 1; //to avoid mod 0
        for(size_t i = 0; i<batch_size; i++){
            size_t start = (cursor+i)%max_start;

            for(size_t t = 0; t < block_size; t++){
                size_t idx = i * block_size + t;
                size_t token_idx = start + t;

                batch.x[idx] = tokens[token_idx];
                batch.y[idx] = tokens[token_idx + 1];
            }
        }

        //move cursor for next call
        cursor = (cursor + batch_size) % max_start;

        return batch;
    }

    size_t CharDataset::get_block_size() const {
        return block_size;
    }

    size_t CharDataset::num_tokens() const {
        return tokens.size();
    }
} //namespace data