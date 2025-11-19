#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"
#include <string>
#include <unordered_map>

// Forward declarations
class Weights;

// Layer normalization
class LayerNorm {
public:
    LayerNorm(const std::string& prefix, const Weights& weights);
    Tensor forward(const Tensor& x) const;
    
private:
    Tensor gamma;  // scale
    Tensor beta;   // bias
    float eps = 1e-5f;
};

// Linear layer (fully connected)
class Linear {
public:
    Linear(const std::string& prefix, const Weights& weights, bool has_bias = true);
    Tensor forward(const Tensor& x) const;
    
private:
    Tensor weight;
    Tensor bias;
    bool m_has_bias;
};

// KV Cache for efficient generation
struct KVCache {
    std::vector<Tensor> keys;    // One per layer
    std::vector<Tensor> values;  // One per layer
    int current_length = 0;
    
    void clear() {
        keys.clear();
        values.clear();
        current_length = 0;
    }
};

// Multi-head attention
class MultiHeadAttention {
public:
    MultiHeadAttention(int n_embd, int n_head, const std::string& prefix, const Weights& weights);
    
    // Forward with optional KV cache
    Tensor forward(const Tensor& x, KVCache* cache = nullptr, int layer_idx = 0) const;
    
private:
    int n_embd;
    int n_head;
    int head_dim;
    
    Linear c_attn;  // Combined QKV projection
    Linear c_proj;  // Output projection
    
    // Split into Q, K, V
    std::tuple<Tensor, Tensor, Tensor> split_qkv(const Tensor& qkv) const;
    
    // Scaled dot-product attention with optional cached K,V
    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v) const;
};

// Feed-forward network (MLP)
class FeedForward {
public:
    FeedForward(int n_embd, const std::string& prefix, const Weights& weights);
    Tensor forward(const Tensor& x) const;
    
private:
    Linear c_fc;    // First layer (n_embd -> 4*n_embd)
    Linear c_proj;  // Second layer (4*n_embd -> n_embd)
    
    Tensor gelu(const Tensor& x) const;
};

// Complete transformer block
class TransformerBlock {
public:
    TransformerBlock(int n_embd, int n_head, int layer_idx, const Weights& weights);
    
    // Forward with optional KV cache
    Tensor forward(const Tensor& x, KVCache* cache = nullptr) const;
    
private:
    int m_layer_idx;
    LayerNorm ln_1;
    MultiHeadAttention attn;
    LayerNorm ln_2;
    FeedForward mlp;
};

// Activation functions
namespace Activations {
    Tensor gelu(const Tensor& x);
    Tensor softmax(const Tensor& x, int dim = -1);
}

#endif // LAYERS_H