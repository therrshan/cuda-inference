#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include "layers.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

// Model configuration from config.json
struct ModelConfig {
    int vocab_size;
    int n_positions;     // context length
    int n_embd;          // embedding dimension
    int n_layer;         // number of layers
    int n_head;          // number of attention heads
    
    // Load from JSON file
    static ModelConfig load(const std::string& config_path);
    void print() const;
};

// Weight storage
class Weights {
public:
    Weights() = default;
    
    // Load weights from binary file
    static Weights load(const std::string& weights_path);
    
    // Get a specific weight tensor
    Tensor get(const std::string& name) const;
    bool has(const std::string& name) const;
    
    // Print weight info
    void print_info() const;
    
    // All weights
    std::unordered_map<std::string, Tensor> tensors;
};

// GPT-2 Model
class GPT2Model {
public:
    GPT2Model(const std::string& weights_dir);
    
    // Forward pass
    Tensor forward(const std::vector<int>& input_ids);
    
    // Text generation
    std::vector<int> generate(const std::vector<int>& input_ids, 
                              int max_new_tokens = 50,
                              float temperature = 1.0f);
    
    // Get model info
    const ModelConfig& config() const { return m_config; }
    
private:
    ModelConfig m_config;
    Weights m_weights;
    Device m_device;
    
    // Model components (will implement later)
    Tensor embedding(const std::vector<int>& input_ids);
    Tensor transformer_block(const Tensor& x, int layer_idx);
    Tensor lm_head(const Tensor& x);
};

#endif // MODEL_H