#pragma once

#include <vector>
#include <cstddef> //for size_t

using namespace std; 

namespace math {
    //a basic 2D matrix of doubles in row-major layour
    //this will be our low-level numeric building block; Tensor will later sit on top of it
    struct Matrix {
        size_t rows;
        size_t cols;
        vector<double> data; //size = rows * cols

        //default consttructor - 0*0 matrix
        Matrix();

        //contruct a matrix with given shape
        Matrix(size_t rows, size_t cols, double default_value = 0); //default value is 0

        //access element at (r, c) - implement bound checking
        double& at(size_t r, size_t c);
        const double& at(size_t r, size_t c) const;

        //total number of element
        size_t size() const;

        //resize matrix to new shape, iotionally filling with some default value
        //this will reallocate data too
        void resize(size_t new_rows, size_t new_cols, double default_value = 0.0);

        //fill all entries with the same value
        void fill(double value);
    };

    Matrix zeros(size_t rows, size_t cols); //create a row x cols filled with zeroes

    Matrix constant(size_t rows, size_t cols, double value); //same idea as zeroes, but for a constant


    //precondiction for add and substract - matrix have same shape
    //add matrix C = A + B
    Matrix add(const Matrix& A, const Matrix& B);


    //subtract C = A-B
    Matrix subtract(const Matrix& A, const Matrix& B);

    //element-wise hadamard product C = A⊙B. again A and B shall have the same shape
    Matrix hadamard(const Matrix& A, const Matrix& B);

    //scalar multiplication
    Matrix scalar_mul(const Matrix& A, double alpha);

    // Standard matrix multiplication: C = A * B
    // Precondition: A.cols == B.rows.
    Matrix matmul(const Matrix& A, const Matrix& B);

    // Transpose of a matrix: B = A^T
    Matrix transpose(const Matrix& A);

    // Compute row-wise softmax.
    // For each row i, softmax is applied over columns: softmax(A[i, :]).
    // Used later for attention scores.
    Matrix row_softmax(const Matrix& A);

    //  pretty-print utility for debugging (definition will use ostream <<).
    void print(const Matrix& A);

} // namespace math
