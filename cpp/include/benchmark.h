#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "inference_engine.h"
#include "tokenizer.h"
#include <string>
#include <chrono>

struct BenchResult {
    double total_seconds;
    double avg_seconds;
    double std_seconds;
    int iterations;
};

class Benchmark {
public:
    // Run simple throughput benchmark: call infer repeatedly and measure time
    static BenchResult throughput(InferenceEngine& engine,
                                  Tokenizer& tokenizer,
                                  const std::string& prompt,
                                  int iterations = 50,
                                  int warmup = 5);

    // Run latency benchmark: measure single-pass latency repeatedly
    static BenchResult latency(InferenceEngine& engine,
                               Tokenizer& tokenizer,
                               const std::string& prompt,
                               int iterations = 200,
                               int warmup = 10);

    // Run generation benchmark: measure time to generate N tokens
    static BenchResult generation(InferenceEngine& engine,
                                  Tokenizer& tokenizer,
                                  const std::string& prompt,
                                  int max_new_tokens = 20,
                                  int iterations = 10,
                                  int warmup = 1);
};

#endif // BENCHMARK_H
