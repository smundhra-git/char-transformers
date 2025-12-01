# N-Dimensional Transformer from Scratch in C++

A lightweight, dependency-free implementation of a Transformer model (based on "Attention Is All You Need") written entirely in C++. 

## Features

- **N-Dimensional Tensor Engine**: Custom autograd engine supporting arbitrary rank tensors (similar to PyTorch).
- **Transformer Architecture**:
  - Multi-Head Self-Attention
  - Sinusoidal Positional Encoding
  - Layer Normalization & Residual Connections
  - Word-level Tokenization
- **Optimization**: Implements Adam Optimizer and SGD.
- **Zero Dependencies**: Pure C++ implementation (no Eigen, no BLAS).

## Build & Run

Ensure you have `cmake` and a C++17 compatible compiler.

```bash
cmake .
make
./main
```

## Structure

- `src/engine`: Autograd and Tensor operations.
- `src/nn`: Neural network modules (Transformer, Attention, Embedding, etc.).
- `src/optim`: Optimizers (Adam, SGD).
- `src/data`: Tokenization and Dataset utilities.

