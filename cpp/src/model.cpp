#include "model.h"
#include "layers.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ==================== ModelConfig ====================

ModelConfig ModelConfig::load(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_path);
    }
    
    json j;
    file >> j;
    
    ModelConfig config;
    config.vocab_size = j["vocab_size"];
    config.n_positions = j["n_positions"];
    config.n_embd = j["n_embd"];
    config.n_layer = j["n_layer"];
    config.n_head = j["n_head"];
    
    return config;
}

void ModelConfig::print() const {
    std::cout << "=== Model Configuration ===" << std::endl;
    std::cout << "Vocab size: " << vocab_size << std::endl;
    std::cout << "Context length: " << n_positions << std::endl;
    std::cout << "Embedding dim: " << n_embd << std::endl;
    std::cout << "Num layers: " << n_layer << std::endl;
    std::cout << "Num heads: " << n_head << std::endl;
}

// ==================== Weights ====================

Weights Weights::load(const std::string& weights_path) {
    std::cout << "Loading weights from: " << weights_path << std::endl;
    
    std::ifstream file(weights_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weights file: " + weights_path);
    }
    
    Weights weights;
    
    // Read number of tensors
    uint32_t num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint32_t));
    std::cout << "Number of tensors: " << num_tensors << std::endl;
    
    for (uint32_t i = 0; i < num_tensors; i++) {
        // Read tensor name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        
        std::vector<char> name_buf(name_len);
        file.read(name_buf.data(), name_len);
        std::string name(name_buf.begin(), name_buf.end());
        
        // Read shape
        uint32_t ndims;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(uint32_t));
        
        std::vector<int> shape(ndims);
        for (uint32_t d = 0; d < ndims; d++) {
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
            shape[d] = static_cast<int>(dim);
        }
        
        // Read dtype
        uint32_t dtype_code;
        file.read(reinterpret_cast<char*>(&dtype_code), sizeof(uint32_t));
        
        // Read data size
        uint64_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
        
        // Read data
        std::vector<char> data_buf(data_size);
        file.read(data_buf.data(), data_size);
        
        // Create tensor
        // For now, assume 2D tensors or flatten to 2D
        int rows = shape[0];
        int cols = (shape.size() == 1) ? 1 : shape[1];
        
        if (shape.size() > 2) {
            // Flatten higher dimensional tensors
            cols = 1;
            for (size_t d = 1; d < shape.size(); d++) {
                cols *= shape[d];
            }
        }
        
        Eigen::MatrixXf matrix(rows, cols);
        std::memcpy(matrix.data(), data_buf.data(), data_size);
        
        Tensor tensor(matrix);
        weights.tensors[name] = tensor;
        
        if (i < 5) {  // Print first 5 tensors
            std::cout << "  " << name << ": shape=[";
            for (size_t d = 0; d < shape.size(); d++) {
                std::cout << shape[d];
                if (d < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    std::cout << "✓ Loaded " << weights.tensors.size() << " tensors" << std::endl;
    return weights;
}

Tensor Weights::get(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

bool Weights::has(const std::string& name) const {
    return tensors.find(name) != tensors.end();
}

void Weights::print_info() const {
    std::cout << "\n=== Loaded Weights ===" << std::endl;
    for (const auto& [name, tensor] : tensors) {
        auto shape = tensor.shape();
        std::cout << name << ": [";
        for (size_t i = 0; i < shape.size(); i++) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

// ==================== GPT2Model ====================

GPT2Model::GPT2Model(const std::string& weights_dir) 
    : m_device(Device::CPU) {
    
    std::cout << "\n=== Initializing GPT-2 Model ===" << std::endl;
    
    // Load config
    std::string config_path = weights_dir + "/config.json";
    m_config = ModelConfig::load(config_path);
    m_config.print();
    
    // Load weights
    std::string weights_path = weights_dir + "/gpt2_weights.bin";
    m_weights = Weights::load(weights_path);
    
    std::cout << "\n✓ Model initialized successfully" << std::endl;
}

Tensor GPT2Model::embedding(const std::vector<int>& input_ids) {
    // Token embeddings
    Tensor wte = m_weights.get("transformer.wte.weight");
    
    // Position embeddings
    Tensor wpe = m_weights.get("transformer.wpe.weight");
    
    int seq_len = input_ids.size();
    
    // Gather token embeddings
    Eigen::MatrixXf token_emb(seq_len, m_config.n_embd);
    for (int i = 0; i < seq_len; i++) {
        token_emb.row(i) = wte.matrix().row(input_ids[i]);
    }
    
    // Add position embeddings
    Eigen::MatrixXf pos_emb(seq_len, m_config.n_embd);
    for (int i = 0; i < seq_len; i++) {
        pos_emb.row(i) = wpe.matrix().row(i);
    }
    
    Eigen::MatrixXf embeddings = token_emb + pos_emb;
    return Tensor(embeddings);
}

Tensor GPT2Model::transformer_block(const Tensor& x, int layer_idx) {
    TransformerBlock block(m_config.n_embd, m_config.n_head, layer_idx, m_weights);
    return block.forward(x);
}

Tensor GPT2Model::lm_head(const Tensor& x) {
    // Final layer norm
    LayerNorm ln_f("transformer.ln_f", m_weights);
    Tensor normalized = ln_f.forward(x);
    
    // Project to vocabulary
    // Note: In GPT-2, lm_head weight is tied to wte (token embedding)
    Tensor wte = m_weights.get("transformer.wte.weight");
    
    // Get last token's hidden state
    Eigen::MatrixXf hidden = normalized.matrix();
    Eigen::VectorXf last_hidden = hidden.row(hidden.rows() - 1);
    
    // Project: last_hidden @ wte^T
    Eigen::VectorXf logits = wte.matrix() * last_hidden;
    
    return Tensor(logits.transpose());  // Shape: [1, vocab_size]
}

Tensor GPT2Model::forward(const std::vector<int>& input_ids) {
    std::cout << "Forward pass with " << input_ids.size() << " tokens" << std::endl;
    
    // Embeddings
    Tensor x = embedding(input_ids);
    std::cout << "  Embeddings: [" << x.shape()[0] << ", " << x.shape()[1] << "]" << std::endl;
    
    // Transformer blocks
    for (int i = 0; i < m_config.n_layer; i++) {
        std::cout << "  Layer " << i << "..." << std::endl;
        x = transformer_block(x, i);
    }
    
    // Language model head
    Tensor logits = lm_head(x);
    std::cout << "  Output logits: [" << logits.shape()[0] << ", " << logits.shape()[1] << "]" << std::endl;
    
    return logits;
}

std::vector<int> GPT2Model::generate(const std::vector<int>& input_ids,
                                      int max_new_tokens,
                                      float temperature) {
    std::cout << "Generating " << max_new_tokens << " tokens..." << std::endl;
    
    std::vector<int> output = input_ids;
    
    for (int i = 0; i < max_new_tokens; i++) {
        // Forward pass
        Tensor logits = forward(output);
        
        // Apply temperature
        Eigen::VectorXf logits_vec = logits.matrix().row(0);
        logits_vec /= temperature;
        
        // Get probabilities with softmax
        Tensor probs_tensor(logits_vec.transpose());
        probs_tensor = Activations::softmax(probs_tensor, -1);
        
        // Greedy sampling: take argmax
        Eigen::VectorXf probs = probs_tensor.matrix().row(0);
        int next_token;
        probs.maxCoeff(&next_token);
        
        output.push_back(next_token);
        
        std::cout << "  Generated token " << i+1 << ": " << next_token << std::endl;
    }
    
    return output;
}