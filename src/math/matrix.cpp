#include "matrix.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <stdexcept> //for throwing errors - love em

namespace math{
    using namespace std;
    //implemention of math

    //first the default constructor
    Matrix::Matrix() : rows(0), cols(0), data(){
        data.clear();
    }

    Matrix::Matrix(size_t r, size_t c, double default_value) : rows(r), cols(c), data(r*c, default_value){
        //fill with default value
    }

    double& Matrix::at(size_t r, size_t c){
        //if values not in bound, throw error
        if(r >= rows || c >= cols){
            throw invalid_argument("row or col out of bounds");
        }
        return data[r*cols+c];
    }

    const double& Matrix::at(size_t r, size_t c) const {
        //if values not in bound, throw error
        //is there something else to do here?
        if(r >= rows || c >= cols){
            throw invalid_argument("row or col out of bounds");
        }
        return data[r*cols+c];
    }

    size_t Matrix::size() const {
        return rows*cols;
    }

    void Matrix::resize(size_t new_rows, size_t new_cols, double default_value) {
        rows = new_rows;
        cols = new_cols;
        data.assign(rows * cols, default_value);
    }
    

    void Matrix::fill(double value) {
        // fill "data" with value
        std::fill(data.begin(), data.end(), value); //using std to not confuse myself?
    }


    Matrix zeros(size_t r, size_t c) {
        Matrix M(r, c, 0);
        return M;
    }

    Matrix constant(size_t r, size_t c, double value) {
        return Matrix(r, c, value);
    }


    Matrix add(const Matrix& A, const Matrix& B) {
        //check matrix rows and cols are the same
        if(A.rows != B.rows || A.cols != B.cols){
            throw invalid_argument("Matrix have different sizes");
        }
        //else add them
        Matrix M(A.rows, A.cols);
        for(size_t i = 0; i<A.size(); i++){
            M.data[i] = A.data[i] + B.data[i];
        }
        return M;
    }

    Matrix subtract(const Matrix& A, const Matrix& B) {
        //check matrix rows and cols are the same
        if(A.rows != B.rows || A.cols != B.cols){
            throw invalid_argument("Matrix have different sizes");
        }
        //else add them
        Matrix M(A.rows, A.cols);
        for(size_t i = 0; i<A.size(); i++){
            M.data[i] = A.data[i] - B.data[i];
        }
        return M;
    }

    Matrix hadamard(const Matrix& A, const Matrix& B) {
        //check matrix rows and cols are the same
        if(A.rows != B.rows || A.cols != B.cols){
            throw invalid_argument("Matrix have different sizes");
        }
        //else add them
        Matrix M(A.rows, A.cols);
        for(size_t i = 0; i<A.size(); i++){
            M.data[i] = A.data[i] * B.data[i];
        }
        return M;
    }

    Matrix scalar_mul(const Matrix& A, double alpha) {
        Matrix M(A.rows, A.cols);
        for(size_t i = 0; i<A.size(); i++){
            M.data[i] = A.data[i] * alpha;
        }
        return M;
    }

    Matrix matmul(const Matrix& A, const Matrix& B) {
        if (A.cols != B.rows) {
            throw invalid_argument("cols of A != rows of B");
        }
        Matrix C(A.rows, B.cols, 0.0);
        for (size_t i = 0; i < A.rows; ++i) {
            for (size_t j = 0; j < B.cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < A.cols; ++k) {
                    sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
                }
                C.data[i * C.cols + j] = sum;
            }
        }
        return C;
    }
    

    // ------------------------
    // Transpose
    // ------------------------

    Matrix transpose(const Matrix& A) {
        Matrix B(A.cols, A.rows, 0.0); // note: cols, rows
        for (size_t i = 0; i < A.rows; ++i) {
            for (size_t j = 0; j < A.cols; ++j) {
                // B(j, i) = A(i, j)
                B.data[j * B.cols + i] = A.data[i * A.cols + j];
            }
        }
        return B;
    }
    


    Matrix row_softmax(const Matrix& A) {
        Matrix result(A.rows, A.cols, 0.0);
    
        for (std::size_t i = 0; i < A.rows; ++i) {
            // 1. find max in row i
            double row_max = A.data[i * A.cols];
            for (std::size_t j = 1; j < A.cols; ++j) {
                double val = A.data[i * A.cols + j];
                if (val > row_max) row_max = val;
            }
    
            // 2. compute exp(x - row_max) and accumulate sum
            double sum = 0.0;
            for (std::size_t j = 0; j < A.cols; ++j) {
                double shifted = A.data[i * A.cols + j] - row_max;
                double e = std::exp(shifted);
                result.data[i * result.cols + j] = e;
                sum += e;
            }
    
            // 3. normalize
            if (sum == 0.0) {
                // edge case: all -inf; here we can either leave zeros or make uniform.
                // For our use case this should basically never happen.
                continue;
            }
    
            for (std::size_t j = 0; j < A.cols; ++j) {
                result.data[i * result.cols + j] /= sum;
            }
        }
    
        return result;
    }
    


    void print(const Matrix& A) {
        for(size_t i = 0; i<A.rows; i++){
            for(size_t j = 0; j<A.cols; j++){
                cout << A.at(i, j) << " ";
            }
            cout << endl;
        }
    }

} // namespace math