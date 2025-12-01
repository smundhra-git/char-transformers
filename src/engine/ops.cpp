#include "ops.hpp"
#include "node.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace engine {
    using namespace std;
    
    // ... [Helpers and other ops] ...
    // I will keep other ops as is, but since I'm rewriting the file, I need to include them.
    // To save context length, I will assume I can use "..." but I cannot. I must write valid C++.
    // I will repeat the file content but update matmul.

    static void check_same_shape(const Tensor& a, const Tensor& b, const char* op) {
        if (a.shape() != b.shape()) {
            throw runtime_error(string(op) + ": shape mismatch");
        }
    }

    struct AddNode : public Node {
        void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            Tensor& b = inputs[1];
            if (a.require_grad()) {
                for(size_t i=0; i<grad_out.size(); ++i) a.grad()[i] += grad_out[i];
            }
            if (b.require_grad()) {
                for(size_t i=0; i<grad_out.size(); ++i) b.grad()[i] += grad_out[i];
            }
        }
    };
    Tensor add(const Tensor& a, const Tensor& b) {
        check_same_shape(a, b, "add");
        Tensor out(a.shape(), a.require_grad() || b.require_grad());
        size_t n = a.numel();
        for(size_t i=0; i<n; ++i) out.data()[i] = a.data()[i] + b.data()[i];
        if(out.require_grad()) {
            auto node = make_shared<AddNode>();
            node->inputs = {a, b};
            out.grad_fn() = node;
        }
        return out;
    }

    struct SubNode : public Node {
        void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            Tensor& b = inputs[1];
            if (a.require_grad()) {
                for(size_t i=0; i<grad_out.size(); ++i) a.grad()[i] += grad_out[i];
            }
            if (b.require_grad()) {
                for(size_t i=0; i<grad_out.size(); ++i) b.grad()[i] -= grad_out[i];
            }
        }
    };
    Tensor sub(const Tensor& a, const Tensor& b) {
        check_same_shape(a, b, "sub");
        Tensor out(a.shape(), a.require_grad() || b.require_grad());
        size_t n = a.numel();
        for(size_t i=0; i<n; ++i) out.data()[i] = a.data()[i] - b.data()[i];
        if(out.require_grad()) {
            auto node = make_shared<SubNode>();
            node->inputs = {a, b};
            out.grad_fn() = node;
        }
        return out;
    }

    struct ScaleNode : public Node {
        double alpha;
        ScaleNode(double a) : alpha(a) {}
        void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            if(a.require_grad()) {
                for(size_t i=0; i<grad_out.size(); ++i) a.grad()[i] += grad_out[i] * alpha;
            }
        }
    };
    Tensor scale(const Tensor& a, double alpha) {
        Tensor out(a.shape(), a.require_grad());
        size_t n = a.numel();
        for(size_t i=0; i<n; ++i) out.data()[i] = a.data()[i] * alpha;
        if(out.require_grad()) {
            auto node = make_shared<ScaleNode>(alpha);
            node->inputs = {a};
            out.grad_fn() = node;
        }
        return out;
    }

    struct MulNode : public Node { 
         void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            Tensor& b = inputs[1];
            const double* a_data = a.data().data();
            const double* b_data = b.data().data();
            if (a.require_grad()) {
                for(size_t i=0; i<grad_out.size(); ++i) a.grad()[i] += grad_out[i] * b_data[i];
            }
            if (b.require_grad()) {
                for(size_t i=0; i<grad_out.size(); ++i) b.grad()[i] += grad_out[i] * a_data[i];
            }
        }
    };
    Tensor mul(const Tensor& a, const Tensor& b) {
        check_same_shape(a, b, "mul");
        Tensor out(a.shape(), a.require_grad() || b.require_grad());
        size_t n = a.numel();
        for(size_t i=0; i<n; ++i) out.data()[i] = a.data()[i] * b.data()[i];
        if(out.require_grad()) {
            auto node = make_shared<MulNode>();
            node->inputs = {a, b};
            out.grad_fn() = node;
        }
        return out;
    }

    // ------------------------------------------------------------------------
    // MatMul (Generalized Batch)
    // ------------------------------------------------------------------------
    struct MatmulNode : public Node {
        void backward(const vector<double>& grad_out) override {
            Tensor& A = inputs[0];
            Tensor& B = inputs[1];
            
            size_t ndimA = A.dim();
            size_t ndimB = B.dim();
            
            // Calc effective Batch
            size_t batchA = 1; for(size_t i=0; i<ndimA-2; ++i) batchA *= A.shape(i);
            size_t batchB = 1; for(size_t i=0; i<ndimB-2; ++i) batchB *= B.shape(i);
            
            bool broadcastB = (batchB == 1 && batchA > 1); // Simplistic broadcast check
            size_t Batch = batchA;

            size_t M = A.shape(ndimA-2);
            size_t K = A.shape(ndimA-1);
            size_t N = B.shape(ndimB-1);

            const double* g_ptr = grad_out.data();
            const double* a_ptr = A.data().data();
            const double* b_ptr = B.data().data();
            
            double* da_ptr = A.require_grad() ? A.grad().data() : nullptr;
            double* db_ptr = B.require_grad() ? B.grad().data() : nullptr;

            size_t stride_A = M * K;
            size_t stride_B = broadcastB ? 0 : (K * N);
            size_t stride_C = M * N;

            for (size_t b = 0; b < Batch; ++b) {
                const double* curr_g = g_ptr + b * stride_C;
                const double* curr_a = a_ptr + b * stride_A;
                const double* curr_b = b_ptr + b * stride_B;
                
                // dA += G @ B^T
                if (da_ptr) {
                    double* curr_da = da_ptr + b * stride_A;
                    for (size_t m = 0; m < M; ++m) {
                        for (size_t n = 0; n < N; ++n) {
                            double g_val = curr_g[m * N + n];
                            for (size_t k = 0; k < K; ++k) {
                                curr_da[m * K + k] += g_val * curr_b[k * N + n];
                            }
                        }
                    }
                }

                // dB += A^T @ G
                if (db_ptr) {
                    // If broadcastB, we accumulate into SAME db buffer
                    double* curr_db = broadcastB ? db_ptr : (db_ptr + b * stride_B);
                    for (size_t m = 0; m < M; ++m) {
                        for (size_t k = 0; k < K; ++k) {
                            double a_val = curr_a[m * K + k];
                            for (size_t n = 0; n < N; ++n) {
                                curr_db[k * N + n] += a_val * curr_g[m * N + n];
                            }
                        }
                    }
                }
            }
        }
    };

    Tensor matmul(const Tensor& A, const Tensor& B) {
        size_t ndimA = A.dim();
        size_t ndimB = B.dim();
        
        if (ndimA < 2 || ndimB < 2) throw runtime_error("matmul: need at least 2D");

        size_t M = A.shape(ndimA-2);
        size_t K = A.shape(ndimA-1);
        if (B.shape(ndimB-2) != K) throw runtime_error("matmul: inner dim mismatch");
        size_t N = B.shape(ndimB-1);

        // Batch calculation
        size_t batchA = 1; for(size_t i=0; i<ndimA-2; ++i) batchA *= A.shape(i);
        size_t batchB = 1; for(size_t i=0; i<ndimB-2; ++i) batchB *= B.shape(i);

        bool broadcastB = false;
        size_t Batch = batchA;
        
        if (batchA == batchB) {
            // Standard batched
        } else if (batchB == 1) {
            broadcastB = true;
        } else {
            throw runtime_error("matmul: batch mismatch (broadcasting only supported for B=1)");
        }

        // Output Shape = A.shape[:-1] + {N}
        Shape out_shape = A.shape();
        out_shape.back() = N;
        
        Tensor C(out_shape, A.require_grad() || B.require_grad());
        
        const double* a_ptr = A.data().data();
        const double* b_ptr = B.data().data();
        double* c_ptr = C.data().data();

        size_t stride_A = M * K;
        size_t stride_B = broadcastB ? 0 : (K * N);
        size_t stride_C = M * N;

        for (size_t b = 0; b < Batch; ++b) {
            const double* curr_a = a_ptr + b * stride_A;
            const double* curr_b = b_ptr + b * stride_B;
            double* curr_c = c_ptr + b * stride_C;

            for (size_t m = 0; m < M; ++m) {
                for (size_t k = 0; k < K; ++k) {
                    double a_val = curr_a[m * K + k];
                    for (size_t n = 0; n < N; ++n) {
                        curr_c[m * N + n] += a_val * curr_b[k * N + n];
                    }
                }
            }
        }

        if (C.require_grad()) {
            auto node = make_shared<MatmulNode>();
            node->inputs = {A, B};
            C.grad_fn() = node;
        }
        return C;
    }

    // Reshape
    struct ReshapeNode : public Node {
        Shape old_shape;
        ReshapeNode(const Shape& s) : old_shape(s) {}
        void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            if(a.require_grad()) {
                for(size_t i=0; i<grad_out.size(); ++i) a.grad()[i] += grad_out[i];
            }
        }
    };
    Tensor reshape(const Tensor& a, const Shape& shape) {
        size_t size = 1;
        for(auto s : shape) size *= s;
        if(size != a.numel()) throw runtime_error("reshape: numel mismatch");
        Tensor out(a.data(), shape, a.require_grad());
        if(out.require_grad()) {
            auto node = make_shared<ReshapeNode>(a.shape());
            node->inputs = {a};
            out.grad_fn() = node;
        }
        return out;
    }

    // Permute
    struct PermuteNode : public Node {
        Shape src_shape;
        vector<size_t> dims; 
        PermuteNode(const Shape& s, const vector<size_t>& d) : src_shape(s), dims(d) {}
        void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            if(!a.require_grad()) return;
            size_t rank = src_shape.size();
            vector<size_t> src_strides(rank);
            size_t s = 1;
            for(int i=rank-1; i>=0; --i) { src_strides[i] = s; s *= src_shape[i]; }
            Shape dst_shape(rank);
            for(size_t i=0; i<rank; ++i) dst_shape[i] = src_shape[dims[i]];
            vector<size_t> dst_strides(rank);
            s = 1;
            for(int i=rank-1; i>=0; --i) { dst_strides[i] = s; s *= dst_shape[i]; }
            size_t numel = a.numel();
            for(size_t i=0; i<numel; ++i) {
                size_t temp = i;
                size_t src_offset = 0;
                for(size_t d=0; d<rank; ++d) {
                    size_t coord = temp / dst_strides[d];
                    temp %= dst_strides[d];
                    size_t src_dim = dims[d];
                    src_offset += coord * src_strides[src_dim];
                }
                a.grad()[src_offset] += grad_out[i];
            }
        }
    };
    Tensor permute(const Tensor& a, const vector<size_t>& dims) {
        size_t rank = a.dim();
        if (dims.size() != rank) throw runtime_error("permute: rank mismatch");
        Shape dst_shape(rank);
        for(size_t i=0; i<rank; ++i) dst_shape[i] = a.shape(dims[i]);
        Tensor out(dst_shape, a.require_grad());
        vector<size_t> src_strides(rank);
        size_t s = 1;
        for(int i=rank-1; i>=0; --i) { src_strides[i] = s; s *= a.shape(i); }
        vector<size_t> dst_strides(rank);
        s = 1;
        for(int i=rank-1; i>=0; --i) { dst_strides[i] = s; s *= dst_shape[i]; }
        const double* src = a.data().data();
        double* dst = out.data().data();
        size_t numel = a.numel();
        for(size_t i=0; i<numel; ++i) {
            size_t temp = i;
            size_t src_offset = 0;
            for(size_t d=0; d<rank; ++d) {
                size_t coord = temp / dst_strides[d];
                temp %= dst_strides[d];
                size_t src_dim = dims[d];
                src_offset += coord * src_strides[src_dim];
            }
            dst[i] = src[src_offset];
        }
        if (out.require_grad()) {
            auto node = make_shared<PermuteNode>(a.shape(), dims);
            node->inputs = {a};
            out.grad_fn() = node;
        }
        return out;
    }
    Tensor transpose(const Tensor& a) {
        size_t rank = a.dim();
        if (rank < 2) throw runtime_error("transpose: need at least 2 dims");
        vector<size_t> dims(rank);
        std::iota(dims.begin(), dims.end(), 0);
        std::swap(dims[rank-1], dims[rank-2]);
        return permute(a, dims);
    }

    // Softmax
    struct SoftmaxNode : public Node {
        int dim;
        SoftmaxNode(int d) : dim(d) {}
        void backward(const vector<double>& grad_out) override {
            Tensor& x = inputs[0];
            if(!x.require_grad()) return;
            size_t rank = x.dim();
            int d = (dim < 0) ? (int)rank + dim : dim;
            size_t inner = 1; 
            for(int i=d+1; i<rank; ++i) inner *= x.shape(i);
            size_t D = x.shape(d);
            size_t outer = x.numel() / (D * inner);
            const double* val = x.data().data();
            double* grad = x.grad().data();
            const double* gout = grad_out.data();
            for(size_t o=0; o<outer; ++o) {
                for(size_t in=0; in<inner; ++in) {
                    size_t base = o * (D * inner) + in;
                    double max_val = -1e9;
                    for(size_t i=0; i<D; ++i) {
                        size_t idx = base + i * inner;
                        if(val[idx] > max_val) max_val = val[idx];
                    }
                    vector<double> y(D);
                    double sum = 0.0;
                    for(size_t i=0; i<D; ++i) {
                        size_t idx = base + i * inner;
                        y[i] = std::exp(val[idx] - max_val);
                        sum += y[i];
                    }
                    for(size_t i=0; i<D; ++i) y[i] /= sum;
                    double dot = 0.0;
                    for(size_t i=0; i<D; ++i) {
                        size_t idx = base + i * inner;
                        dot += y[i] * gout[idx];
                    }
                    for(size_t i=0; i<D; ++i) {
                        size_t idx = base + i * inner;
                        grad[idx] += y[i] * (gout[idx] - dot);
                    }
                }
            }
        }
    };
    Tensor softmax(const Tensor& x, int dim) {
        size_t rank = x.dim();
        int d = (dim < 0) ? (int)rank + dim : dim;
        if (d < 0 || d >= rank) throw runtime_error("softmax: dim out of range");
        Tensor out(x.shape(), x.require_grad());
        size_t inner = 1; 
        for(int i=d+1; i<rank; ++i) inner *= x.shape(i);
        size_t D = x.shape(d);
        size_t outer = x.numel() / (D * inner);
        const double* src = x.data().data();
        double* dst = out.data().data();
        for(size_t o=0; o<outer; ++o) {
            for(size_t in=0; in<inner; ++in) {
                size_t base = o * (D * inner) + in;
                double max_val = -1e9;
                for(size_t i=0; i<D; ++i) {
                    size_t idx = base + i * inner;
                    if(src[idx] > max_val) max_val = src[idx];
                }
                double sum = 0.0;
                for(size_t i=0; i<D; ++i) {
                    size_t idx = base + i * inner;
                    double e = std::exp(src[idx] - max_val);
                    dst[idx] = e;
                    sum += e;
                }
                for(size_t i=0; i<D; ++i) {
                    size_t idx = base + i * inner;
                    dst[idx] /= sum;
                }
            }
        }
        if(out.require_grad()) {
            auto node = make_shared<SoftmaxNode>(d);
            node->inputs = {x};
            out.grad_fn() = node;
        }
        return out;
    }

    // Cross Entropy
    struct CrossEntropyNode : public Node {
        Tensor target; 
        CrossEntropyNode(const Tensor& tgt) : target(tgt) {}
        void backward(const vector<double>& grad_out) override {
            Tensor& logits = inputs[0];
            if(!logits.require_grad()) return;
            double g = grad_out[0]; 
            size_t B = logits.shape(0);
            size_t T = logits.shape(1);
            size_t V = logits.shape(2);
            size_t N = B * T;
            const double* logit_data = logits.data().data();
            double* logit_grad = logits.grad().data();
            const double* tgt_data = target.data().data();
            for (size_t i = 0; i < N; ++i) {
                const double* z = logit_data + i * V;
                double* dz = logit_grad + i * V;
                int label = (int)tgt_data[i];
                double max_val = -1e9;
                for(size_t j=0; j<V; ++j) if(z[j] > max_val) max_val = z[j];
                double sum = 0.0;
                vector<double> p(V);
                for(size_t j=0; j<V; ++j) {
                    p[j] = std::exp(z[j] - max_val);
                    sum += p[j];
                }
                double scale = g / (double)N;
                for(size_t j=0; j<V; ++j) {
                    p[j] /= sum;
                    double y = (j == label) ? 1.0 : 0.0;
                    dz[j] += (p[j] - y) * scale;
                }
            }
        }
    };
    Tensor cross_entropy(const Tensor& logits, const Tensor& target) {
        if (logits.dim() != 3 || target.dim() != 2) 
            throw runtime_error("cross_entropy: expected (B,T,V) and (B,T)");
        if (logits.shape(0) != target.shape(0) || logits.shape(1) != target.shape(1))
            throw runtime_error("cross_entropy: batch/time mismatch");
        size_t B = logits.shape(0);
        size_t T = logits.shape(1);
        size_t V = logits.shape(2);
        size_t N = B * T;
        const double* l_ptr = logits.data().data();
        const double* t_ptr = target.data().data();
        double total_loss = 0.0;
        for (size_t i = 0; i < N; ++i) {
            const double* z = l_ptr + i * V;
            int label = (int)t_ptr[i];
            if (label < 0 || label >= (int)V) throw runtime_error("cross_entropy: label out of bounds");
            double max_val = -1e9;
            for(size_t j=0; j<V; ++j) if(z[j] > max_val) max_val = z[j];
            double sum_exp = 0.0;
            for(size_t j=0; j<V; ++j) sum_exp += std::exp(z[j] - max_val);
            double log_sum_exp = std::log(sum_exp) + max_val;
            total_loss += (log_sum_exp - z[label]);
        }
        double avg_loss = total_loss / N;
        Tensor out({1}, logits.require_grad());
        out.data()[0] = avg_loss;
        if (out.require_grad()) {
            auto node = make_shared<CrossEntropyNode>(target);
            node->inputs = {logits}; 
            out.grad_fn() = node;
        }
        return out;
    }

    // Layer Norm
    struct LayerNormNode : public Node {
        double eps;
        LayerNormNode(double e) : eps(e) {}
        void backward(const vector<double>& grad_out) override {
            Tensor& x = inputs[0];
            Tensor& gamma = inputs[1];
            Tensor& beta = inputs[2];
            if (!x.require_grad() && !gamma.require_grad() && !beta.require_grad()) return;
            size_t D = x.shape().back();
            size_t N = x.numel() / D;
            const double* x_data = x.data().data();
            const double* gam_data = gamma.data().data();
            const double* gout = grad_out.data();
            double* dx = x.require_grad() ? x.grad().data() : nullptr;
            double* dgam = gamma.require_grad() ? gamma.grad().data() : nullptr;
            double* dbeta = beta.require_grad() ? beta.grad().data() : nullptr;
            for (size_t i = 0; i < N; ++i) {
                const double* xi = x_data + i * D;
                const double* gi = gout + i * D;
                double* dxi = dx ? (dx + i * D) : nullptr;
                double mean = 0.0;
                for(size_t j=0; j<D; ++j) mean += xi[j];
                mean /= D;
                double var = 0.0;
                for(size_t j=0; j<D; ++j) var += (xi[j]-mean)*(xi[j]-mean);
                var /= D;
                double std_inv = 1.0 / std::sqrt(var + eps);
                for(size_t j=0; j<D; ++j) {
                    double x_norm = (xi[j] - mean) * std_inv;
                    if (dgam) dgam[j] += gi[j] * x_norm;
                    if (dbeta) dbeta[j] += gi[j];
                }
                if (dxi) {
                    double sum_dxhat = 0.0;
                    double sum_dxhat_xhat = 0.0;
                    for(size_t j=0; j<D; ++j) {
                        double dx_hat = gi[j] * gam_data[j];
                        double x_norm = (xi[j] - mean) * std_inv;
                        sum_dxhat += dx_hat;
                        sum_dxhat_xhat += dx_hat * x_norm;
                    }
                    for(size_t j=0; j<D; ++j) {
                        double dx_hat = gi[j] * gam_data[j];
                        double x_norm = (xi[j] - mean) * std_inv;
                        double term = D * dx_hat - sum_dxhat - x_norm * sum_dxhat_xhat;
                        dxi[j] += (std_inv / D) * term;
                    }
                }
            }
        }
    };
    Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps) {
        size_t D = x.shape().back();
        if (gamma.numel() != D || beta.numel() != D) throw runtime_error("layer_norm: shape mismatch");
        Tensor out(x.shape(), x.require_grad() || gamma.require_grad() || beta.require_grad());
        size_t N = x.numel() / D;
        const double* x_ptr = x.data().data();
        const double* g_ptr = gamma.data().data();
        const double* b_ptr = beta.data().data();
        double* out_ptr = out.data().data();
        for (size_t i = 0; i < N; ++i) {
            const double* row = x_ptr + i * D;
            double* out_row = out_ptr + i * D;
            double mean = 0.0;
            for (size_t j = 0; j < D; ++j) mean += row[j];
            mean /= D;
            double var = 0.0;
            for (size_t j = 0; j < D; ++j) var += (row[j] - mean) * (row[j] - mean);
            var /= D;
            double std_inv = 1.0 / std::sqrt(var + eps);
            for (size_t j = 0; j < D; ++j) {
                out_row[j] = ((row[j] - mean) * std_inv) * g_ptr[j] + b_ptr[j];
            }
        }
        if (out.require_grad()) {
            auto node = make_shared<LayerNormNode>(eps);
            node->inputs = {x, gamma, beta};
            out.grad_fn() = node;
        }
        return out;
    }

    Tensor gelu(const Tensor& x) {
        Tensor out(x.shape(), x.require_grad());
        double k = std::sqrt(2.0 / 3.14159265358979323846);
        size_t n = x.numel();
        for(size_t i=0; i<n; ++i) {
            double val = x.data()[i];
            double cube = val * val * val;
            double inner = k * (val + 0.044715 * cube);
            out.data()[i] = 0.5 * val * (1.0 + std::tanh(inner));
        }
        return out;
    }
}
