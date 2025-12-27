#ifndef LLAMA_WRAPPER_H
#define LLAMA_WRAPPER_H

#include <string>
#include <vector>
#include <functional>
#include <memory>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;
struct llama_sampler;

namespace llama_wrapper {

struct ModelParams {
    std::string model_path;
    int n_gpu_layers = 99;  // Use GPU by default if available
    bool use_mmap = true;
    bool use_mlock = false;
};

struct ContextParams {
    int n_ctx = 2048;      // Context size
    int n_batch = 512;     // Batch size for prompt processing
    int n_threads = 4;     // Number of threads
};

struct GenerationParams {
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    std::vector<std::string> stop_sequences;
};

struct GenerationResult {
    std::string text;
    int prompt_tokens;
    int completion_tokens;
    std::string finish_reason;  // "stop", "length", or "error"
};

// Token callback for streaming: returns false to stop generation
using TokenCallback = std::function<bool(const std::string& token)>;

class LlamaModel {
public:
    LlamaModel();
    ~LlamaModel();

    // Disable copy
    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;

    // Enable move
    LlamaModel(LlamaModel&& other) noexcept;
    LlamaModel& operator=(LlamaModel&& other) noexcept;

    // Load a model from a GGUF file
    bool load(const ModelParams& params);

    // Check if model is loaded
    bool is_loaded() const;

    // Unload the model
    void unload();

    // Get the model path
    const std::string& get_model_path() const { return model_path_; }

    // Create a context for inference
    bool create_context(const ContextParams& params);

    // Generate text (non-streaming)
    GenerationResult generate(const std::string& prompt, const GenerationParams& params);

    // Generate text (streaming)
    GenerationResult generate_streaming(
        const std::string& prompt,
        const GenerationParams& params,
        TokenCallback callback
    );

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    std::string model_path_;

    // Tokenize a string
    std::vector<int32_t> tokenize(const std::string& text, bool add_bos);

    // Detokenize a single token
    std::string detokenize(int32_t token);

    // Create sampler with given params
    void create_sampler(const GenerationParams& params);

    // Check if token is end-of-sequence
    bool is_eos_token(int32_t token);
};

} // namespace llama_wrapper

#endif // LLAMA_WRAPPER_H

