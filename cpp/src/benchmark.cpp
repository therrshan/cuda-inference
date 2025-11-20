#include "benchmark.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

using high_res_clock = std::chrono::high_resolution_clock;

static double seconds_since(const std::chrono::time_point<high_res_clock>& t) {
    return std::chrono::duration<double>(high_res_clock::now() - t).count();
}

static BenchResult compute_stats(const std::vector<double>& samples) {
    BenchResult r{};
    r.iterations = static_cast<int>(samples.size());
    r.total_seconds = std::accumulate(samples.begin(), samples.end(), 0.0);
    r.avg_seconds = r.total_seconds / r.iterations;
    double var = 0.0;
    for (double s : samples) var += (s - r.avg_seconds) * (s - r.avg_seconds);
    var /= r.iterations;
    r.std_seconds = std::sqrt(var);
    return r;
}

BenchResult Benchmark::throughput(InferenceEngine& engine, Tokenizer& tokenizer,
                                  const std::string& prompt, int iterations, int warmup) {
    std::vector<int> ids = tokenizer.encode(prompt);
    std::vector<double> samples;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        engine.infer(ids);
    }

    for (int i = 0; i < iterations; ++i) {
        auto t0 = high_res_clock::now();
        engine.infer(ids);
        double dt = seconds_since(t0);
        samples.push_back(dt);
        std::cout << "[throughput] iter=" << (i+1) << " time=" << dt << "s\n";
    }

    return compute_stats(samples);
}

BenchResult Benchmark::latency(InferenceEngine& engine, Tokenizer& tokenizer,
                               const std::string& prompt, int iterations, int warmup) {
    // Latency: measure time for a single-token forward (simulate incremental)
    std::vector<int> ids = tokenizer.encode(prompt);
    std::vector<double> samples;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        engine.infer(ids);
    }

    for (int i = 0; i < iterations; ++i) {
        auto t0 = high_res_clock::now();
        engine.infer(ids);
        double dt = seconds_since(t0);
        samples.push_back(dt);
        if ((i+1) % 10 == 0) std::cout << "[latency] iter=" << (i+1) << " time=" << dt << "s\n";
    }

    return compute_stats(samples);
}

BenchResult Benchmark::generation(InferenceEngine& engine, Tokenizer& tokenizer,
                                  const std::string& prompt, int max_new_tokens, int iterations, int warmup) {
    std::vector<double> samples;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        engine.generate(tokenizer.encode(prompt), 1, 1.0f);
    }

    for (int i = 0; i < iterations; ++i) {
        auto t0 = high_res_clock::now();
        engine.generate(tokenizer.encode(prompt), max_new_tokens, 1.0f);
        double dt = seconds_since(t0);
        samples.push_back(dt);
        std::cout << "[generation] iter=" << (i+1) << " time=" << dt << "s\n";
    }

    return compute_stats(samples);
}
