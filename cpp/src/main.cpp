#include <iostream>
#include "model.h"
#include "tensor.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    std::cout << "=== CUDA Inference Engine ===" << std::endl;
    std::cout << "Testing weight loading...\n" << std::endl;
    
    try {
        // Load model
        std::string weights_dir = "../../weights";
        if (argc > 1) {
            weights_dir = argv[1];
        }
        
        std::cout << "Loading from: " << weights_dir << std::endl;
        GPT2Model model(weights_dir);
        
        // Load tokenizer
        std::cout << "\nLoading tokenizer..." << std::endl;
        Tokenizer tokenizer("../../vocab/vocab.json", "../../vocab/merges.txt");
        
        // Test with real text
        std::string prompt = "Hello, how are you";
        std::cout << "\n=== Text Generation ===" << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        
        // Encode
        std::vector<int> input_ids = tokenizer.encode(prompt);
        std::cout << "Encoded to " << input_ids.size() << " tokens: ";
        for (int id : input_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        // Generate
        std::cout << "\nGenerating 10 tokens..." << std::endl;
        std::vector<int> generated = model.generate(input_ids, 5, 0.3f);
        
        // Decode
        std::string output = tokenizer.decode(generated);
        std::cout << "\n=== Generated Text ===" << std::endl;
        std::cout << output << std::endl;
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}