#include "tokenizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <algorithm>

using json = nlohmann::json;

Tokenizer::Tokenizer(const std::string& vocab_path, const std::string& merges_path) {
    // Initialize byte-level mappings
    init_byte_mappings();
    
    // Load vocabulary
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        throw std::runtime_error("Cannot open vocab file: " + vocab_path);
    }
    
    json vocab_json;
    vocab_file >> vocab_json;
    
    for (auto& [token, id] : vocab_json.items()) {
        vocab[token] = id;
        id_to_vocab[id] = token;
    }
    
    std::cout << "Loaded vocabulary: " << vocab.size() << " tokens" << std::endl;
    
    // Load BPE merges
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        throw std::runtime_error("Cannot open merges file: " + merges_path);
    }
    
    std::string line;
    int rank = 0;
    while (std::getline(merges_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string first, second;
        iss >> first >> second;
        
        if (!first.empty() && !second.empty()) {
            bpe_ranks[{first, second}] = rank++;
        }
    }
    
    std::cout << "Loaded BPE merges: " << bpe_ranks.size() << std::endl;
}

void Tokenizer::init_byte_mappings() {
    // GPT-2 byte-to-unicode mapping
    std::vector<int> bs;
    
    // Add printable ASCII excluding space (33-126)
    for (int b = 33; b <= 126; b++) bs.push_back(b);
    // Add extended ASCII (161-172, 174-255)  
    for (int b = 161; b <= 172; b++) bs.push_back(b);
    for (int b = 174; b <= 255; b++) bs.push_back(b);
    
    std::vector<int> cs = bs;
    int n = 0;
    
    // Map remaining bytes (including space=32) to unicode 256+
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    
    // Create bidirectional mapping
    for (size_t i = 0; i < bs.size(); i++) {
        byte_encoder[bs[i]] = std::string(1, static_cast<char>(cs[i]));
        byte_decoder[std::string(1, static_cast<char>(cs[i]))] = bs[i];
    }
}

std::vector<unsigned char> Tokenizer::text_to_bytes(const std::string& text) {
    std::vector<unsigned char> bytes;
    for (char c : text) {
        bytes.push_back(static_cast<unsigned char>(c));
    }
    return bytes;
}

std::string Tokenizer::bytes_to_text(const std::vector<unsigned char>& bytes) {
    std::string text;
    for (unsigned char b : bytes) {
        text += byte_encoder[b];
    }
    return text;
}

std::set<std::pair<int, std::pair<std::string, std::string>>> 
Tokenizer::get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<int, std::pair<std::string, std::string>>> pairs;
    
    if (word.size() < 2) return pairs;
    
    for (size_t i = 0; i < word.size() - 1; i++) {
        std::pair<std::string, std::string> pair = {word[i], word[i + 1]};
        
        auto it = bpe_ranks.find(pair);
        if (it != bpe_ranks.end()) {
            pairs.insert(std::make_pair(it->second, pair));
        }
    }
    
    return pairs;
}

std::vector<std::string> Tokenizer::bpe(const std::string& token) {
    if (token.size() <= 1) {
        return {token};
    }
    
    // Split into characters
    std::vector<std::string> word;
    for (char c : token) {
        word.push_back(std::string(1, c));
    }
    
    // Iteratively merge pairs
    while (word.size() > 1) {
        auto pairs = get_pairs(word);
        
        if (pairs.empty()) break;
        
        // Get the pair with lowest rank (highest priority)
        auto bigram = pairs.begin()->second;
        
        // Merge this pair in the word
        std::vector<std::string> new_word;
        size_t i = 0;
        
        while (i < word.size()) {
            if (i < word.size() - 1 && word[i] == bigram.first && word[i + 1] == bigram.second) {
                new_word.push_back(bigram.first + bigram.second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        
        word = new_word;
        
        if (word.size() == 1) break;
    }
    
    return word;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> token_ids;
    
    // Simple word splitting - just split on spaces for now
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // Add space prefix (GPT-2 convention)
        std::string prefixed = " " + word;
        
        // Convert to byte-level
        auto bytes = text_to_bytes(prefixed);
        std::string byte_encoded = bytes_to_text(bytes);
        
        // Apply BPE
        std::vector<std::string> bpe_tokens = bpe(byte_encoded);
        
        // Convert to IDs
        for (const auto& bpe_token : bpe_tokens) {
            auto it = vocab.find(bpe_token);
            if (it != vocab.end()) {
                token_ids.push_back(it->second);
            }
        }
    }
    
    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) {
    std::string byte_string;
    
    // Get byte-encoded tokens
    for (int id : token_ids) {
        auto it = id_to_vocab.find(id);
        if (it != id_to_vocab.end()) {
            byte_string += it->second;
        }
    }
    
    // Decode from byte-level back to UTF-8
    std::vector<unsigned char> bytes;
    for (size_t i = 0; i < byte_string.length(); i++) {
        unsigned char c = static_cast<unsigned char>(byte_string[i]);
        
        // Look up the original byte value
        std::string char_str(1, static_cast<char>(c));
        auto it = byte_decoder.find(char_str);
        if (it != byte_decoder.end()) {
            bytes.push_back(static_cast<unsigned char>(it->second));
        } else {
            // Fallback
            bytes.push_back(c);
        }
    }
    
    // Convert bytes to string
    std::string result;
    for (unsigned char b : bytes) {
        result += static_cast<char>(b);
    }
    
    return result;
}

std::string Tokenizer::id_to_token(int id) const {
    auto it = id_to_vocab.find(id);
    if (it != id_to_vocab.end()) {
        return it->second;
    }
    return "<unk>";
}

int Tokenizer::token_to_id(const std::string& token) const {
    auto it = vocab.find(token);
    if (it != vocab.end()) {
        return it->second;
    }
    return -1;
}