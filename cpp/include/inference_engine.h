#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include "model.h"
#include "tokenizer.h"
#include <string>

// Simple inference engine wrapper around GPT2Model
class InferenceEngine {
public:
    InferenceEngine(const std::string& weights_dir);

    // Run a full forward pass and return logits
    Tensor infer(const std::vector<int>& input_ids);

    // Generate tokens using the underlying model (simple wrapper)
    std::vector<int> generate(const std::vector<int>& input_ids,
                              int max_new_tokens = 50,
                              float temperature = 1.0f);

    // Convenience: encode a prompt and generate text
    std::string generate_from_prompt(const std::string& prompt,
                                     Tokenizer& tokenizer,
                                     int max_new_tokens = 50,
                                     float temperature = 1.0f);

    const ModelConfig& config() const { return m_model.config(); }

private:
    GPT2Model m_model;
};

#endif // INFERENCE_ENGINE_H
