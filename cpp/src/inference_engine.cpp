#include "inference_engine.h"
#include <iostream>

InferenceEngine::InferenceEngine(const std::string& weights_dir)
    : m_model(weights_dir) {
    std::cout << "InferenceEngine initialized with weights: " << weights_dir << std::endl;
}

Tensor InferenceEngine::infer(const std::vector<int>& input_ids) {
    return m_model.forward(input_ids);
}

std::vector<int> InferenceEngine::generate(const std::vector<int>& input_ids,
                                           int max_new_tokens,
                                           float temperature) {
    return m_model.generate(input_ids, max_new_tokens, temperature);
}

std::string InferenceEngine::generate_from_prompt(const std::string& prompt,
                                                  Tokenizer& tokenizer,
                                                  int max_new_tokens,
                                                  float temperature) {
    std::vector<int> input_ids = tokenizer.encode(prompt);
    std::vector<int> out_ids = generate(input_ids, max_new_tokens, temperature);
    return tokenizer.decode(out_ids);
}
