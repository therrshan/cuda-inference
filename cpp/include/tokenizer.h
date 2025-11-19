#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <set>

// Hash function for pair keys (needs to be defined before use)
struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
    }
};

class Tokenizer {
public:
    Tokenizer(const std::string& vocab_path, const std::string& merges_path);
    
    // Encode text to token IDs
    std::vector<int> encode(const std::string& text);
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& token_ids);
    
    // Get single token
    std::string id_to_token(int id) const;
    int token_to_id(const std::string& token) const;
    
    int vocab_size() const { return vocab.size(); }
    
private:
    std::unordered_map<std::string, int> vocab;  // token -> id
    std::unordered_map<int, std::string> id_to_vocab;  // id -> token
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks;
    
    // Byte-level encoding (GPT-2 style)
    std::unordered_map<int, std::string> byte_encoder;
    std::unordered_map<std::string, int> byte_decoder;
    
    // BPE helpers
    std::vector<std::string> bpe(const std::string& token);
    std::set<std::pair<int, std::pair<std::string, std::string>>> get_pairs(const std::vector<std::string>& word);
    
    // Initialize byte encoder/decoder
    void init_byte_mappings();
    std::string bytes_to_text(const std::vector<unsigned char>& bytes);
    std::vector<unsigned char> text_to_bytes(const std::string& text);
};

#endif // TOKENIZER_H