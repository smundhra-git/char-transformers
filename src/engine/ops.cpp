#include "ops.hpp"
#include "node.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

namespace engine {
    using namespace std;
    
    static Tensor ensure_contiguous(const Tensor& t) {
        if (!t.is_contiguous()) {
            return t.contiguous();
        }
        return t;
    }

    static void check_same_shape(const Tensor& a, const Tensor& b, const char* op) {
        if (a.shape() != b.shape()) {
            throw runtime_error(string(op) + ": shape mismatch");
        }
    }

    struct AddNode : public Node {
        void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            Tensor& b = inputs[1];
            size_t n = a.numel(); 
            
            if (a.require_grad()) {
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for(size_t i=0; i<n; ++i) a.grad()[a.offset() + i] += grad_out[i]; // Fix: Add offset (though grad usually offset 0 for now)
            }
            if (b.require_grad()) {
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for(size_t i=0; i<n; ++i) b.grad()[b.offset() + i] += grad_out[i];
            }
        }
    };
    Tensor add(const Tensor& a, const Tensor& b) {
        Tensor ac = ensure_contiguous(a);
        Tensor bc = ensure_contiguous(b);
        
        check_same_shape(ac, bc, "add");
        Tensor out(ac.shape(), ac.require_grad() || bc.require_grad());
        size_t n = ac.numel();
        
        const double* ad = ac.data().data() + ac.offset();
        const double* bd = bc.data().data() + bc.offset();
        double* od = out.data().data() + out.offset();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(size_t i=0; i<n; ++i) od[i] = ad[i] + bd[i];
        
        if(out.require_grad()) {
            auto node = make_shared<AddNode>();
            node->inputs = {ac, bc};
            out.grad_fn() = node;
        }
        return out;
    }

    struct SubNode : public Node {
        void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            Tensor& b = inputs[1];
            size_t n = a.numel();
            if (a.require_grad()) {
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for(size_t i=0; i<n; ++i) a.grad()[a.offset() + i] += grad_out[i];
            }
            if (b.require_grad()) {
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for(size_t i=0; i<n; ++i) b.grad()[b.offset() + i] -= grad_out[i];
            }
        }
    };
    Tensor sub(const Tensor& a, const Tensor& b) {
        Tensor ac = ensure_contiguous(a);
        Tensor bc = ensure_contiguous(b);
        check_same_shape(ac, bc, "sub");
        Tensor out(ac.shape(), ac.require_grad() || bc.require_grad());
        size_t n = ac.numel();
        
        const double* ad = ac.data().data() + ac.offset();
        const double* bd = bc.data().data() + bc.offset();
        double* od = out.data().data() + out.offset();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(size_t i=0; i<n; ++i) od[i] = ad[i] - bd[i];

        if(out.require_grad()) {
            auto node = make_shared<SubNode>();
            node->inputs = {ac, bc};
            out.grad_fn() = node;
        }
        return out;
    }

    struct ScaleNode : public Node {
        double alpha;
        ScaleNode(double a) : alpha(a) {}
        void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            size_t n = a.numel();
            if(a.require_grad()) {
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for(size_t i=0; i<n; ++i) a.grad()[a.offset() + i] += grad_out[i] * alpha;
            }
        }
    };
    Tensor scale(const Tensor& a, double alpha) {
        Tensor ac = ensure_contiguous(a);
        Tensor out(ac.shape(), ac.require_grad());
        size_t n = ac.numel();
        
        const double* ad = ac.data().data() + ac.offset();
        double* od = out.data().data() + out.offset();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(size_t i=0; i<n; ++i) od[i] = ad[i] * alpha;

        if(out.require_grad()) {
            auto node = make_shared<ScaleNode>(alpha);
            node->inputs = {ac};
            out.grad_fn() = node;
        }
        return out;
    }

    struct MulNode : public Node { 
         void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            Tensor& b = inputs[1];
            const double* a_data = a.data().data() + a.offset();
            const double* b_data = b.data().data() + b.offset();
            size_t n = a.numel();

            if (a.require_grad()) {
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for(size_t i=0; i<n; ++i) a.grad()[a.offset() + i] += grad_out[i] * b_data[i];
            }
            if (b.require_grad()) {
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for(size_t i=0; i<n; ++i) b.grad()[b.offset() + i] += grad_out[i] * a_data[i];
            }
        }
    };
    Tensor mul(const Tensor& a, const Tensor& b) {
        Tensor ac = ensure_contiguous(a);
        Tensor bc = ensure_contiguous(b);
        check_same_shape(ac, bc, "mul");
        Tensor out(ac.shape(), ac.require_grad() || bc.require_grad());
        size_t n = ac.numel();
        
        const double* ad = ac.data().data() + ac.offset();
        const double* bd = bc.data().data() + bc.offset();
        double* od = out.data().data() + out.offset();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(size_t i=0; i<n; ++i) od[i] = ad[i] * bd[i];

        if(out.require_grad()) {
            auto node = make_shared<MulNode>();
            node->inputs = {ac, bc};
            out.grad_fn() = node;
        }
        return out;
    }

    struct MatmulNode : public Node {
        void backward(const vector<double>& grad_out) override {
            Tensor& A = inputs[0];
            Tensor& B = inputs[1];
            
            size_t ndimA = A.dim();
            size_t ndimB = B.dim();
            
            size_t batchA = 1; for(size_t i=0; i<ndimA-2; ++i) batchA *= A.shape(i);
            size_t batchB = 1; for(size_t i=0; i<ndimB-2; ++i) batchB *= B.shape(i);
            
            bool broadcastB = (batchB == 1 && batchA > 1);
            size_t Batch = batchA;

            size_t M = A.shape(ndimA-2);
            size_t K = A.shape(ndimA-1);
            size_t N = B.shape(ndimB-1);

            const double* g_ptr = grad_out.data(); // grad_out is passed directly (usually vector&)
            // Note: Node::backward assumes grad_out is contiguous vector.
            // If A/B are views, we need their offsets.
            
            const double* a_ptr = A.data().data() + A.offset();
            const double* b_ptr = B.data().data() + B.offset();
            
            double* da_ptr = A.require_grad() ? (A.grad().data() + A.offset()) : nullptr;
            double* db_ptr = B.require_grad() ? (B.grad().data() + B.offset()) : nullptr;

            size_t stride_A = M * K;
            size_t stride_B = broadcastB ? 0 : (K * N);
            size_t stride_C = M * N;

            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (size_t b = 0; b < Batch; ++b) {
                const double* curr_g = g_ptr + b * stride_C;
                const double* curr_a = a_ptr + b * stride_A;
                const double* curr_b = b_ptr + b * stride_B;
                
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

                if (db_ptr) {
                    if (broadcastB) {
                        #ifdef _OPENMP
                        #pragma omp critical
                        #endif
                        {
                            double* curr_db = db_ptr; 
                            for (size_t m = 0; m < M; ++m) {
                                for (size_t k = 0; k < K; ++k) {
                                    double a_val = curr_a[m * K + k];
                                    for (size_t n = 0; n < N; ++n) {
                                        curr_db[k * N + n] += a_val * curr_g[m * N + n];
                                    }
                                }
                            }
                        }
                    } else {
                        double* curr_db = db_ptr + b * stride_B;
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
        }
    };

    Tensor matmul(const Tensor& A, const Tensor& B) {
        Tensor ac = ensure_contiguous(A);
        Tensor bc = ensure_contiguous(B);
        
        size_t ndimA = ac.dim();
        size_t ndimB = bc.dim();
        
        if (ndimA < 2 || ndimB < 2) throw runtime_error("matmul: need at least 2D");

        size_t M = ac.shape(ndimA-2);
        size_t K = ac.shape(ndimA-1);
        if (bc.shape(ndimB-2) != K) throw runtime_error("matmul: inner dim mismatch");
        size_t N = bc.shape(ndimB-1);

        size_t batchA = 1; for(size_t i=0; i<ndimA-2; ++i) batchA *= ac.shape(i);
        size_t batchB = 1; for(size_t i=0; i<ndimB-2; ++i) batchB *= bc.shape(i);

        bool broadcastB = false;
        size_t Batch = batchA;
        
        if (batchA == batchB) {
        } else if (batchB == 1) {
            broadcastB = true;
        } else {
            throw runtime_error("matmul: batch mismatch");
        }

        Shape out_shape = ac.shape();
        out_shape.back() = N;
        
        Tensor C(out_shape, ac.require_grad() || bc.require_grad());
        
        const double* a_ptr = ac.data().data() + ac.offset();
        const double* b_ptr = bc.data().data() + bc.offset();
        double* c_ptr = C.data().data() + C.offset();

        size_t stride_A = M * K;
        size_t stride_B = broadcastB ? 0 : (K * N);
        size_t stride_C = M * N;

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
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
            node->inputs = {ac, bc};
            C.grad_fn() = node;
        }
        return C;
    }

    struct ReshapeNode : public Node {
        Shape old_shape;
        ReshapeNode(const Shape& s) : old_shape(s) {}
        void backward(const vector<double>& grad_out) override {
            Tensor& a = inputs[0];
            // Check if grad storage is distinct (copy happened) or shared (view)
            // If shared, grad_out IS a.grad(), so nothing to do.
            // If copied, we accumulate.
            // How to check? Pointer comparison.
            const double* ga = a.grad().data(); // Storage start
            const double* go = grad_out.data(); // This is likely output's storage start if passed from backward() loop
            
            // But wait, backward() loop passes t.grad().
            // If 'a' and 'output' share storage, t.grad() == a.grad().
            // So ga == go.
            // However, we have offsets.
            
            // In the case of copy, storage is different.
            if (a.grad().data() != grad_out.data()) { // Crude check for different storage
                 size_t n = grad_out.size(); // This is whole storage size? No, grad_out passed as vector ref.
                 // Vector ref doesn't know about tensor view size.
                 // We must trust `numel()` matching.
                 // Actually `backward()` passes `t.grad()`, which is the WHOLE storage vector.
                 
                 // If we did a COPY reshape, `grad_out` corresponds to the COPY's storage.
                 // We need to add it to `a`'s storage.
                 // But mapping indices back is hard if strides involved.
                 // For contiguous reshape copy, it's linear.
                 
                 if(a.require_grad()) {
                     size_t num = a.numel();
                     // Assuming compact linear copy
                     #ifdef _OPENMP
                     #pragma omp parallel for
                     #endif
                     for(size_t i=0; i<num; ++i) a.grad()[a.offset() + i] += grad_out[i]; // Assumes grad_out offset 0
                 }
            }
        }
    };
    Tensor reshape(const Tensor& a, const Shape& shape) {
        Tensor out = a.reshape(shape);
        if(out.require_grad()) {
            auto node = make_shared<ReshapeNode>(a.shape());
            node->inputs = {a};
            out.grad_fn() = node;
        }
        return out;
    }

    struct PermuteNode : public Node {
        Shape src_shape;
        vector<size_t> dims; 
        PermuteNode(const Shape& s, const vector<size_t>& d) : src_shape(s), dims(d) {}
        void backward(const vector<double>& grad_out) override {
             // View operation: grad shared usually.
             // If not shared (fallback copy?), simplistic engine might fail here.
             // Assuming zero-copy view -> grads shared -> no-op.
        }
    };
    Tensor permute(const Tensor& a, const vector<size_t>& dims) {
        size_t rank = a.dim();
        if (dims.size() != rank) throw runtime_error("permute: rank mismatch");
        
        Shape dst_shape(rank);
        vector<size_t> dst_strides(rank);
        const auto& src_strides = a.strides();

        for(size_t i=0; i<rank; ++i) {
            dst_shape[i] = a.shape(dims[i]);
            dst_strides[i] = src_strides[dims[i]];
        }

        Tensor out(a.p->storage, a.offset(), dst_shape, dst_strides, a.require_grad());

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
            
            const double* val = x.data().data() + x.offset();
            double* grad = x.grad().data() + x.offset();
            const double* gout = grad_out.data(); // Assumed contiguous from output

            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
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
        Tensor xc = ensure_contiguous(x);
        size_t rank = xc.dim();
        int d = (dim < 0) ? (int)rank + dim : dim;
        Tensor out(xc.shape(), xc.require_grad());
        
        size_t inner = 1; 
        for(int i=d+1; i<rank; ++i) inner *= xc.shape(i);
        size_t D = xc.shape(d);
        size_t outer = xc.numel() / (D * inner);
        
        const double* src = xc.data().data() + xc.offset();
        double* dst = out.data().data() + out.offset();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
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
            node->inputs = {xc};
            out.grad_fn() = node;
        }
        return out;
    }

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
            const double* logit_data = logits.data().data() + logits.offset();
            double* logit_grad = logits.grad().data() + logits.offset();
            const double* tgt_data = target.data().data() + target.offset();

            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (size_t i = 0; i < N; ++i) {
                const double* z = logit_data + i * V;
                double* dz = logit_grad + i * V;
                int label = (int)tgt_data[i];
                
                double max_val = -1e9;
                for(size_t j=0; j<V; ++j) if(z[j] > max_val) max_val = z[j];
                
                double sum = 0.0;
                for(size_t j=0; j<V; ++j) sum += std::exp(z[j] - max_val);
                
                double scale = g / (double)N;
                for(size_t j=0; j<V; ++j) {
                    double p = std::exp(z[j] - max_val) / sum;
                    double y = (j == label) ? 1.0 : 0.0;
                    dz[j] += (p - y) * scale;
                }
            }
        }
    };
    Tensor cross_entropy(const Tensor& logits, const Tensor& target) {
        Tensor lc = ensure_contiguous(logits);
        Tensor tc = ensure_contiguous(target);
        size_t B = lc.shape(0);
        size_t T = lc.shape(1);
        size_t V = lc.shape(2);
        size_t N = B * T;
        
        const double* l_ptr = lc.data().data() + lc.offset();
        const double* t_ptr = tc.data().data() + tc.offset();
        
        double total_loss = 0.0;
        
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:total_loss)
        #endif
        for (size_t i = 0; i < N; ++i) {
            const double* z = l_ptr + i * V;
            int label = (int)t_ptr[i];
            
            double max_val = -1e9;
            for(size_t j=0; j<V; ++j) if(z[j] > max_val) max_val = z[j];
            
            double sum_exp = 0.0;
            for(size_t j=0; j<V; ++j) sum_exp += std::exp(z[j] - max_val);
            
            double log_sum_exp = std::log(sum_exp) + max_val;
            total_loss += (log_sum_exp - z[label]);
        }

        double avg_loss = total_loss / N;
        Tensor out({1}, lc.require_grad());
        out.data()[0] = avg_loss;
        if (out.require_grad()) {
            auto node = make_shared<CrossEntropyNode>(tc);
            node->inputs = {lc}; 
            out.grad_fn() = node;
        }
        return out;
    }

    struct LayerNormNode : public Node {
        double eps;
        LayerNormNode(double e) : eps(e) {}
        void backward(const vector<double>& grad_out) override {
            Tensor& x = inputs[0];
            Tensor& gamma = inputs[1];
            Tensor& beta = inputs[2];
            
            size_t D = x.shape().back();
            size_t N = x.numel() / D;
            
            const double* x_data = x.data().data() + x.offset();
            const double* gam_data = gamma.data().data() + gamma.offset();
            const double* gout = grad_out.data(); // Assumed output contiguous
            
            double* dx = x.require_grad() ? (x.grad().data() + x.offset()) : nullptr;
            double* dgam = gamma.require_grad() ? (gamma.grad().data() + gamma.offset()) : nullptr;
            double* dbeta = beta.require_grad() ? (beta.grad().data() + beta.offset()) : nullptr;

            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
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
                    if (dgam) { 
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        dgam[j] += gi[j] * x_norm; 
                    }
                    if (dbeta) {
                        #ifdef _OPENMP
                        #pragma omp atomic
                        #endif
                        dbeta[j] += gi[j];
                    }
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
        Tensor xc = ensure_contiguous(x);
        size_t D = xc.shape().back();
        Tensor out(xc.shape(), xc.require_grad() || gamma.require_grad() || beta.require_grad());
        size_t N = xc.numel() / D;
        
        const double* x_ptr = xc.data().data() + xc.offset();
        const double* g_ptr = gamma.data().data() + gamma.offset();
        const double* b_ptr = beta.data().data() + beta.offset();
        double* out_ptr = out.data().data() + out.offset();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
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
            node->inputs = {xc, gamma, beta};
            out.grad_fn() = node;
        }
        return out;
    }

    Tensor gelu(const Tensor& x) {
        Tensor xc = ensure_contiguous(x);
        Tensor out(xc.shape(), xc.require_grad());
        double k = std::sqrt(2.0 / 3.14159265358979323846);
        size_t n = xc.numel();
        
        const double* inp = xc.data().data() + xc.offset();
        double* outp = out.data().data() + out.offset();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(size_t i=0; i<n; ++i) {
            double val = inp[i];
            double cube = val * val * val;
            double inner = k * (val + 0.044715 * cube);
            outp[i] = 0.5 * val * (1.0 + std::tanh(inner));
        }
        return out;
    }
}
