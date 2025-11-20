#include "layers.h"
#include "model.h"
#include <cmath>
#include <iostream>

// ==================== Activation Functions ====================

Tensor Activations::gelu(const Tensor& x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    
    Eigen::MatrixXf x_mat = x.matrix();
    Eigen::MatrixXf x_cubed = x_mat.array().cube();
    Eigen::ArrayXXf inner = sqrt_2_over_pi * (x_mat.array() + 0.044715f * x_cubed.array());
    Eigen::MatrixXf result = 0.5f * x_mat.array() * (1.0f + inner.tanh());
    
    return Tensor(result);
}

Tensor Activations::softmax(const Tensor& x, int dim) {
    Eigen::MatrixXf x_mat = x.matrix();
    
    if (dim == -1 || dim == 1) {
        // Softmax across columns (each row independently)
        Eigen::MatrixXf result(x_mat.rows(), x_mat.cols());
        
        for (int i = 0; i < x_mat.rows(); i++) {
            float max_val = x_mat.row(i).maxCoeff();
            Eigen::VectorXf exp_row = (x_mat.row(i).array() - max_val).exp();
            float sum = exp_row.sum();
            result.row(i) = exp_row / sum;
        }
        return Tensor(result);
    } else {
        // Softmax across rows
        Eigen::MatrixXf result(x_mat.rows(), x_mat.cols());
        
        for (int j = 0; j < x_mat.cols(); j++) {
            float max_val = x_mat.col(j).maxCoeff();
            Eigen::VectorXf exp_col = (x_mat.col(j).array() - max_val).exp();
            float sum = exp_col.sum();
            result.col(j) = exp_col / sum;
        }
        return Tensor(result);
    }
}

// ==================== LayerNorm ====================

LayerNorm::LayerNorm(const std::string& prefix, const Weights& weights) {
    gamma = weights.get(prefix + ".weight");
    beta = weights.get(prefix + ".bias");
}

Tensor LayerNorm::forward(const Tensor& x) const {
    Eigen::MatrixXf x_mat = x.matrix();
    Eigen::MatrixXf result(x_mat.rows(), x_mat.cols());
    
    // Normalize each row independently
    for (int i = 0; i < x_mat.rows(); i++) {
        float mean = x_mat.row(i).mean();
        float variance = (x_mat.row(i).array() - mean).square().mean();
        float std_dev = std::sqrt(variance + eps);
        
        // Normalize
        Eigen::VectorXf normalized = (x_mat.row(i).array() - mean) / std_dev;
        
        // Scale and shift
        result.row(i) = normalized.array() * gamma.matrix().array() + beta.matrix().array();
    }
    
    return Tensor(result);
}

// ==================== Linear ====================

Linear::Linear(const std::string& prefix, const Weights& weights, bool has_bias)
    : m_has_bias(has_bias) {
    
    weight = weights.get(prefix + ".weight");
    if (has_bias) {
        bias = weights.get(prefix + ".bias");
    }
}

Tensor Linear::forward(const Tensor& x) const {
    // PyTorch weights are stored as (out_features, in_features) in state_dict
    // But when we loaded them, they might be transposed
    // x: [seq_len, in_features]
    // weight: Check actual shape and do the right operation
    
    // Debug prints removed to reduce noisy output during forward passes
    
    Tensor result;
    
    // If weight is [in_features, out_features], do x @ W
    if (weight.shape()[0] == x.shape()[1]) {
        result = x.matmul(weight);
    }
    // If weight is [out_features, in_features], do x @ W^T
    else if (weight.shape()[1] == x.shape()[1]) {
        result = x.matmul(weight.transpose());
    }
    else {
        throw std::runtime_error("Weight dimensions don't match input");
    }
    
    if (m_has_bias) {
        // Broadcast bias across batch dimension
        Eigen::MatrixXf result_mat = result.matrix();
        Eigen::VectorXf bias_vec;
        
        // Handle bias shape
        if (bias.shape()[0] == 1) {
            bias_vec = bias.matrix().row(0);
        } else {
            bias_vec = bias.matrix().col(0);
        }
        
        for (int i = 0; i < result_mat.rows(); i++) {
            result_mat.row(i) += bias_vec.transpose();
        }
        result = Tensor(result_mat);
    }
    
    return result;
}

// ==================== MultiHeadAttention ====================

MultiHeadAttention::MultiHeadAttention(int n_embd, int n_head, 
                                       const std::string& prefix, 
                                       const Weights& weights)
    : n_embd(n_embd), 
      n_head(n_head),
      head_dim(n_embd / n_head),
      c_attn(prefix + ".c_attn", weights),
      c_proj(prefix + ".c_proj", weights) {
}

std::tuple<Tensor, Tensor, Tensor> MultiHeadAttention::split_qkv(const Tensor& qkv) const {
    // qkv shape: [batch_size, seq_len, 3 * n_embd]
    // Split into Q, K, V each of shape [batch_size, seq_len, n_embd]
    
    Eigen::MatrixXf qkv_mat = qkv.matrix();
    int seq_len = qkv_mat.rows();
    
    Eigen::MatrixXf q_mat = qkv_mat.block(0, 0, seq_len, n_embd);
    Eigen::MatrixXf k_mat = qkv_mat.block(0, n_embd, seq_len, n_embd);
    Eigen::MatrixXf v_mat = qkv_mat.block(0, 2 * n_embd, seq_len, n_embd);
    
    return {Tensor(q_mat), Tensor(k_mat), Tensor(v_mat)};
}

Tensor MultiHeadAttention::attention(const Tensor& q, const Tensor& k, const Tensor& v) const {
    // Simplified single-head attention for now
    // scores = (Q @ K^T) / sqrt(d_k)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    Tensor scores = q.matmul(k.transpose()).mul(scale);
    
    // Apply softmax
    Tensor attn_weights = Activations::softmax(scores, -1);
    
    // Apply attention to values
    Tensor output = attn_weights.matmul(v);
    
    return output;
}

Tensor MultiHeadAttention::forward(const Tensor& x, KVCache* cache, int layer_idx) const {
    // Project to Q, K, V
    Tensor qkv = c_attn.forward(x);
    
    // Split into Q, K, V
    auto [q, k, v] = split_qkv(qkv);
    
    // If we have a cache, concatenate cached K,V with new K,V
    if (cache && !cache->keys.empty() && layer_idx < (int)cache->keys.size()) {
        // Concatenate along sequence dimension
        Eigen::MatrixXf k_cached = cache->keys[layer_idx].matrix();
        Eigen::MatrixXf v_cached = cache->values[layer_idx].matrix();
        Eigen::MatrixXf k_new = k.matrix();
        Eigen::MatrixXf v_new = v.matrix();
        
        // Concatenate: [cached; new]
        Eigen::MatrixXf k_full(k_cached.rows() + k_new.rows(), k_cached.cols());
        k_full << k_cached, k_new;
        
        Eigen::MatrixXf v_full(v_cached.rows() + v_new.rows(), v_cached.cols());
        v_full << v_cached, v_new;
        
        k = Tensor(k_full);
        v = Tensor(v_full);
    }
    
    // Update cache with new K,V
    if (cache) {
        if (layer_idx >= (int)cache->keys.size()) {
            cache->keys.resize(layer_idx + 1);
            cache->values.resize(layer_idx + 1);
        }
        cache->keys[layer_idx] = k;
        cache->values[layer_idx] = v;
    }
    
    // Apply attention
    Tensor attn_output = attention(q, k, v);
    
    // Project output
    Tensor output = c_proj.forward(attn_output);
    
    return output;
}

// ==================== FeedForward ====================

FeedForward::FeedForward(int n_embd, const std::string& prefix, const Weights& weights)
    : c_fc(prefix + ".c_fc", weights),
      c_proj(prefix + ".c_proj", weights) {
}

Tensor FeedForward::forward(const Tensor& x) const {
    // First layer + GELU
    Tensor hidden = c_fc.forward(x);
    hidden = Activations::gelu(hidden);
    
    // Second layer
    Tensor output = c_proj.forward(hidden);
    
    return output;
}

// ==================== TransformerBlock ====================

TransformerBlock::TransformerBlock(int n_embd, int n_head, int layer_idx, const Weights& weights)
    : m_layer_idx(layer_idx),
      ln_1("transformer.h." + std::to_string(layer_idx) + ".ln_1", weights),
      attn(n_embd, n_head, "transformer.h." + std::to_string(layer_idx) + ".attn", weights),
      ln_2("transformer.h." + std::to_string(layer_idx) + ".ln_2", weights),
      mlp(n_embd, "transformer.h." + std::to_string(layer_idx) + ".mlp", weights) {
}

Tensor TransformerBlock::forward(const Tensor& x, KVCache* cache) const {
    // Self-attention with residual
    Tensor attn_output = attn.forward(ln_1.forward(x), cache, m_layer_idx);
    Tensor x_after_attn = x.add(attn_output);
    
    // Feed-forward with residual
    Tensor mlp_output = mlp.forward(ln_2.forward(x_after_attn));
    Tensor output = x_after_attn.add(mlp_output);
    
    return output;
}