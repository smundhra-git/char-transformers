#include <iostream>
#include <string>
#include <vector>

#include "data/vocab.hpp"
#include "data/dataset.hpp"

using namespace std;
using namespace data;

int main() {
    cout << "=== Vocab + CharDataset test ===" << endl;

    // Tiny toy corpus
    string corpus = "hello\nworld";

    // 1) Build vocab
    Vocab vocab;
    vocab.build_from_text(corpus);

    cout << "Vocab built. Size = " << vocab.size() << endl;

    // 2) Encode / decode test
    vector<int> ids = vocab.encode("hello");
    cout << "Encoded 'hello' -> ";
    for (int id : ids) cout << id << " ";
    cout << endl;

    string decoded = vocab.decode(ids);
    cout << "Decoded back -> '" << decoded << "'" << endl;

    // 3) Full corpus as tokens
    vector<int> tokens = vocab.encode(corpus);
    cout << "Corpus token length = " << tokens.size() << endl;

    // 4) Dataset test
    size_t block_size = 4;
    size_t batch_size = 2;

    CharDataset dataset(tokens, block_size);

    Batch batch = dataset.next_batch(batch_size);

    cout << "\nBatch.x (ids):" << endl;
    for (size_t i = 0; i < batch.batch_size; ++i) {
        cout << "row " << i << ": ";
        for (size_t t = 0; t < batch.block_size; ++t) {
            int id = batch.x[i * batch.block_size + t];
            cout << id << " ";
        }
        cout << endl;
    }

    cout << "\nBatch.x (decoded):" << endl;
    for (size_t i = 0; i < batch.batch_size; ++i) {
        string row;
        for (size_t t = 0; t < batch.block_size; ++t) {
            int id = batch.x[i * batch.block_size + t];
            row.push_back(vocab.decode({id})[0]);
        }
        cout << "row " << i << ": '" << row << "'" << endl;
    }

    cout << "\nBatch.y (decoded):" << endl;
    for (size_t i = 0; i < batch.batch_size; ++i) {
        string row;
        for (size_t t = 0; t < batch.block_size; ++t) {
            int id = batch.y[i * batch.block_size + t];
            row.push_back(vocab.decode({id})[0]);
        }
        cout << "row " << i << ": '" << row << "'" << endl;
    }

    return 0;
}
