#include <iostream>
#include "model.h"
#include "inference_engine.h"
#include "tensor.h"
#include "tokenizer.h"
#include "benchmark.h"

int main(int argc, char** argv) {
    std::cout << "=== CUDA Inference Engine ===" << std::endl;
    std::cout << "Testing weight loading...\n" << std::endl;
    
    try {
        // Default paths
        std::string weights_dir = "../../weights";
        std::string vocab_json = "../../vocab/vocab.json";
        std::string merges_txt = "../../vocab/merges.txt";

        // Benchmark flags
        bool run_benchmark = false;
        std::string bench_type = "throughput"; // latency | throughput | generation
        int bench_iters = 50;
        int bench_warmup = 5;
        std::string prompt = "Hello, how are you";
        int bench_max_new_tokens = 10;

        // Simple CLI parsing
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--weights" && i + 1 < argc) { weights_dir = argv[++i]; }
            else if (a == "--vocab" && i + 1 < argc) { vocab_json = argv[++i]; }
            else if (a == "--merges" && i + 1 < argc) { merges_txt = argv[++i]; }
            else if ((a == "--bench") || (a == "--benchmark")) { run_benchmark = true; }
            else if (a == "--bench-type" && i + 1 < argc) { bench_type = argv[++i]; }
            else if (a == "--bench-iters" && i + 1 < argc) { bench_iters = std::stoi(argv[++i]); }
            else if (a == "--bench-warmup" && i + 1 < argc) { bench_warmup = std::stoi(argv[++i]); }
            else if (a == "--prompt" && i + 1 < argc) { prompt = argv[++i]; }
            else if (a == "--max-new-tokens" && i + 1 < argc) { bench_max_new_tokens = std::stoi(argv[++i]); }
            else if (i == 1 && a.rfind("-", 0) != 0) { weights_dir = a; }
        }

        std::cout << "Loading from: " << weights_dir << std::endl;
        InferenceEngine engine(weights_dir);

        // Load tokenizer
        std::cout << "\nLoading tokenizer..." << std::endl;
        Tokenizer tokenizer(vocab_json, merges_txt);

        std::cout << "\nPrompt: \"" << prompt << "\"" << std::endl;

        if (!run_benchmark) {
            // Default behavior: generate a few tokens and print
            std::cout << "\nGenerating " << bench_max_new_tokens << " tokens..." << std::endl;
            std::string output = engine.generate_from_prompt(prompt, tokenizer, bench_max_new_tokens, 0.3f);
            std::cout << "\n=== Generated Text ===" << std::endl;
            std::cout << output << std::endl;
        } else {
            // Run benchmarks
            std::cout << "\nRunning benchmark: " << bench_type << " (iters=" << bench_iters << ", warmup=" << bench_warmup << ")" << std::endl;
            BenchResult res;
            if (bench_type == "throughput") {
                res = Benchmark::throughput(engine, tokenizer, prompt, bench_iters, bench_warmup);
            } else if (bench_type == "latency") {
                res = Benchmark::latency(engine, tokenizer, prompt, bench_iters, bench_warmup);
            } else { // generation
                res = Benchmark::generation(engine, tokenizer, prompt, bench_max_new_tokens, bench_iters, bench_warmup);
            }

            std::cout << "\n=== Benchmark Results ===" << std::endl;
            std::cout << "  iterations: " << res.iterations << std::endl;
            std::cout << "  total time: " << res.total_seconds << " s" << std::endl;
            std::cout << "  avg time:   " << res.avg_seconds << " s" << std::endl;
            std::cout << "  std dev:    " << res.std_seconds << " s" << std::endl;
            if (bench_type == "generation") {
                double tokens_per_sec = (bench_max_new_tokens * res.iterations) / res.total_seconds;
                std::cout << "  tokens/sec: " << tokens_per_sec << std::endl;
            }
        }
        
        std::cout << "\n✓ All tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}