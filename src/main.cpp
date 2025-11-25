#include "math/matrix.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;
using namespace math;

// Helper to compare two doubles with tolerance
bool almost_equal(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

// Helper to check that all rows of a matrix sum to ~1 (for softmax)
void assert_rows_sum_to_one(const Matrix& M, double eps = 1e-9) {
    for (std::size_t i = 0; i < M.rows; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < M.cols; ++j) {
            row_sum += M.data[i * M.cols + j];
        }
        assert(almost_equal(row_sum, 1.0, eps));
    }
}

int main() {
    cout << "=== Matrix tests start ===" << endl;

    // 1. Test constructors, zeros, constant, fill, size
    {
        Matrix A(2, 3, 1.5);
        assert(A.rows == 2);
        assert(A.cols == 3);
        assert(A.size() == 6);
        for (std::size_t i = 0; i < A.size(); ++i) {
            assert(almost_equal(A.data[i], 1.5));
        }

        Matrix Z = zeros(2, 2);
        assert(Z.rows == 2 && Z.cols == 2);
        for (std::size_t i = 0; i < Z.size(); ++i) {
            assert(almost_equal(Z.data[i], 0.0));
        }

        Matrix C = constant(3, 1, 7.0);
        assert(C.rows == 3 && C.cols == 1);
        for (std::size_t i = 0; i < C.size(); ++i) {
            assert(almost_equal(C.data[i], 7.0));
        }

        A.fill(0.25);
        for (std::size_t i = 0; i < A.size(); ++i) {
            assert(almost_equal(A.data[i], 0.25));
        }

        cout << "Constructor / zeros / constant / fill tests passed." << endl;
    }

    // 2. Test add, subtract, hadamard, scalar_mul
    {
        Matrix A(2, 2, 0.0);
        A.data = {1.0, 2.0,
                  3.0, 4.0};

        Matrix B(2, 2, 0.0);
        B.data = {5.0, 6.0,
                  7.0, 8.0};

        Matrix S = add(A, B);       // A + B
        Matrix D = subtract(B, A);  // B - A
        Matrix H = hadamard(A, B);  // A .* B
        Matrix M = scalar_mul(A, 2.0);

        // Check A + B
        assert(almost_equal(S.data[0], 6.0));  // 1+5
        assert(almost_equal(S.data[3], 12.0)); // 4+8

        // Check B - A
        assert(almost_equal(D.data[0], 4.0));  // 5-1
        assert(almost_equal(D.data[3], 4.0));  // 8-4

        // Check Hadamard
        assert(almost_equal(H.data[0], 5.0));   // 1*5
        assert(almost_equal(H.data[3], 32.0));  // 4*8

        // Check scalar_mul
        assert(almost_equal(M.data[0], 2.0));   // 1*2
        assert(almost_equal(M.data[3], 8.0));   // 4*2

        cout << "add / subtract / hadamard / scalar_mul tests passed." << endl;
    }

    // 3. Test matmul with known result
    {
        // A: 2x3
        Matrix A(2, 3, 0.0);
        A.data = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};

        // B: 3x2
        Matrix B(3, 2, 0.0);
        B.data = {7.0,  8.0,
                  9.0, 10.0,
                  11.0, 12.0};

        // Expected C = A * B:
        // [58,  64]
        // [139, 154]
        Matrix C = matmul(A, B);

        assert(C.rows == 2);
        assert(C.cols == 2);
        assert(almost_equal(C.data[0], 58.0));
        assert(almost_equal(C.data[1], 64.0));
        assert(almost_equal(C.data[2], 139.0));
        assert(almost_equal(C.data[3], 154.0));

        cout << "matmul test passed." << endl;
    }

    // 4. Test transpose on non-square matrix
    {
        Matrix A(2, 3, 0.0);
        A.data = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};

        Matrix T = transpose(A);
        // T should be 3x2:
        // [1, 4]
        // [2, 5]
        // [3, 6]
        assert(T.rows == 3);
        assert(T.cols == 2);
        assert(almost_equal(T.data[0], 1.0)); // (0,0)
        assert(almost_equal(T.data[1], 4.0)); // (0,1)
        assert(almost_equal(T.data[2], 2.0)); // (1,0)
        assert(almost_equal(T.data[3], 5.0)); // (1,1)
        assert(almost_equal(T.data[4], 3.0)); // (2,0)
        assert(almost_equal(T.data[5], 6.0)); // (2,1)

        cout << "transpose test passed." << endl;
    }

    // 5. Test row_softmax on simple cases
    {
        // Single row [0, 0, 0] -> uniform [1/3, 1/3, 1/3]
        Matrix A(1, 3, 0.0);
        A.data = {0.0, 0.0, 0.0};

        Matrix S = row_softmax(A);
        assert(S.rows == 1);
        assert(S.cols == 3);
        assert_rows_sum_to_one(S);
        // Each should be roughly 1/3
        double p0 = S.data[0];
        double p1 = S.data[1];
        double p2 = S.data[2];
        assert(almost_equal(p0, 1.0 / 3.0, 1e-6));
        assert(almost_equal(p1, 1.0 / 3.0, 1e-6));
        assert(almost_equal(p2, 1.0 / 3.0, 1e-6));

        // Row [1, 2, 3] -> distribution skewed towards 3
        Matrix B(1, 3, 0.0);
        B.data = {1.0, 2.0, 3.0};
        Matrix S2 = row_softmax(B);
        assert_rows_sum_to_one(S2);
        double q0 = S2.data[0];
        double q1 = S2.data[1];
        double q2 = S2.data[2];
        // We don't assert exact values, but the last should be largest
        assert(q2 > q1 && q1 > q0);

        cout << "row_softmax tests passed." << endl;
    }

    // 6. Optional: show a printed example
    {
        Matrix A(2, 3, 0.0);
        A.data = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};
        cout << "Example matrix A:" << endl;
        print(A);
    }

    cout << "=== All Matrix tests passed successfully ===" << endl;
    return 0;
}
