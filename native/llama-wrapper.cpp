#include "llama-wrapper.h"
#include "llama.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace llama_wrapper {

LlamaModel::LlamaModel() = default;

LlamaModel::~LlamaModel() {
    unload();
}

LlamaModel::LlamaModel(LlamaModel&& other) noexcept
    : model_(other.model_)
    , ctx_(other.ctx_)
    , sampler_(other.sampler_)
    , model_path_(std::move(other.model_path_)) {
    other.model_ = nullptr;
    other.ctx_ = nullptr;
    other.sampler_ = nullptr;
}

LlamaModel& LlamaModel::operator=(LlamaModel&& other) noexcept {
    if (this != &other) {
        unload();
        model_ = other.model_;
        ctx_ = other.ctx_;
        sampler_ = other.sampler_;
        model_path_ = std::move(other.model_path_);
        other.model_ = nullptr;
        other.ctx_ = nullptr;
        other.sampler_ = nullptr;
    }
    return *this;
}

bool LlamaModel::load(const ModelParams& params) {
    if (model_) {
        unload();
    }

    // Initialize llama backend
    llama_backend_init();

    // Set up model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    model_params.use_mmap = params.use_mmap;
    model_params.use_mlock = params.use_mlock;

    // Load the model
    model_ = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model_) {
        return false;
    }

    model_path_ = params.model_path;
    return true;
}

bool LlamaModel::is_loaded() const {
    return model_ != nullptr;
}

void LlamaModel::unload() {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
        llama_backend_free();
    }
    model_path_.clear();
}

bool LlamaModel::create_context(const ContextParams& params) {
    if (!model_) {
        return false;
    }

    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx;
    ctx_params.n_batch = params.n_batch;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;

    ctx_ = llama_init_from_model(model_, ctx_params);
    return ctx_ != nullptr;
}

void LlamaModel::create_sampler(const GenerationParams& params) {
    if (sampler_) {
        llama_sampler_free(sampler_);
    }

    // Create a sampler chain
    sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Add samplers to the chain
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(params.top_p, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(params.temperature));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(42)); // Random seed
}

std::vector<int32_t> LlamaModel::tokenize(const std::string& text, bool add_bos) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);

    // First, get the number of tokens needed
    // When passing 0 for n_tokens_max, llama_tokenize returns negative of required size
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, add_bos, true);

    if (n_tokens < 0) {
        n_tokens = -n_tokens;  // Convert to positive size
    }

    if (n_tokens == 0) {
        return {};  // Empty input
    }

    std::vector<int32_t> tokens(n_tokens);
    int actual_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_bos, true);

    if (actual_tokens < 0) {
        // Buffer still too small, resize and try again
        tokens.resize(-actual_tokens);
        actual_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_bos, true);
    }

    if (actual_tokens > 0) {
        tokens.resize(actual_tokens);
    } else {
        tokens.clear();
    }

    return tokens;
}

std::string LlamaModel::detokenize(int32_t token) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);

    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return "";
    }
    return std::string(buf, n);
}

bool LlamaModel::is_eos_token(int32_t token) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    return llama_vocab_is_eog(vocab, token);
}

GenerationResult LlamaModel::generate(const std::string& prompt, const GenerationParams& params) {
    GenerationResult result;
    result.finish_reason = "error";

    if (!ctx_ || !model_) {
        return result;
    }

    // Tokenize the prompt
    std::vector<int32_t> prompt_tokens = tokenize(prompt, true);
    result.prompt_tokens = prompt_tokens.size();
    result.completion_tokens = 0;

    // Clear the memory/KV cache
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // Create sampler
    create_sampler(params);

    // Create batch for prompt processing
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_decode(ctx_, batch) != 0) {
        return result;
    }

    // Generate tokens
    std::string generated_text;
    int n_cur = prompt_tokens.size();

    for (int i = 0; i < params.max_tokens; i++) {
        // Sample the next token
        int32_t new_token = llama_sampler_sample(sampler_, ctx_, -1);

        // Check for end of sequence
        if (is_eos_token(new_token)) {
            result.finish_reason = "stop";
            break;
        }

        // Convert token to string
        std::string token_str = detokenize(new_token);
        generated_text += token_str;
        result.completion_tokens++;

        // Check for stop sequences
        bool should_stop = false;
        for (const auto& stop_seq : params.stop_sequences) {
            if (generated_text.length() >= stop_seq.length()) {
                if (generated_text.substr(generated_text.length() - stop_seq.length()) == stop_seq) {
                    // Remove the stop sequence from output
                    generated_text = generated_text.substr(0, generated_text.length() - stop_seq.length());
                    should_stop = true;
                    result.finish_reason = "stop";
                    break;
                }
            }
        }
        if (should_stop) break;

        // Prepare for next iteration
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx_, batch) != 0) {
            break;
        }
        n_cur++;
    }

    if (result.finish_reason == "error" && result.completion_tokens >= params.max_tokens) {
        result.finish_reason = "length";
    } else if (result.finish_reason == "error") {
        result.finish_reason = "stop";
    }

    result.text = generated_text;
    return result;
}

GenerationResult LlamaModel::generate_streaming(
    const std::string& prompt,
    const GenerationParams& params,
    TokenCallback callback
) {
    GenerationResult result;
    result.finish_reason = "error";

    if (!ctx_ || !model_) {
        return result;
    }

    // Tokenize the prompt
    std::vector<int32_t> prompt_tokens = tokenize(prompt, true);
    result.prompt_tokens = prompt_tokens.size();
    result.completion_tokens = 0;

    // Clear the memory/KV cache
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // Create sampler
    create_sampler(params);

    // Create batch for prompt processing
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_decode(ctx_, batch) != 0) {
        return result;
    }

    // Generate tokens
    std::string generated_text;
    int n_cur = prompt_tokens.size();

    for (int i = 0; i < params.max_tokens; i++) {
        // Sample the next token
        int32_t new_token = llama_sampler_sample(sampler_, ctx_, -1);

        // Check for end of sequence
        if (is_eos_token(new_token)) {
            result.finish_reason = "stop";
            break;
        }

        // Convert token to string
        std::string token_str = detokenize(new_token);
        generated_text += token_str;
        result.completion_tokens++;

        // Call the callback with the new token
        if (!callback(token_str)) {
            result.finish_reason = "stop";
            break;
        }

        // Check for stop sequences
        bool should_stop = false;
        for (const auto& stop_seq : params.stop_sequences) {
            if (generated_text.length() >= stop_seq.length()) {
                if (generated_text.substr(generated_text.length() - stop_seq.length()) == stop_seq) {
                    should_stop = true;
                    result.finish_reason = "stop";
                    break;
                }
            }
        }
        if (should_stop) break;

        // Prepare for next iteration
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx_, batch) != 0) {
            break;
        }
        n_cur++;
    }

    if (result.finish_reason == "error" && result.completion_tokens >= params.max_tokens) {
        result.finish_reason = "length";
    } else if (result.finish_reason == "error") {
        result.finish_reason = "stop";
    }

    result.text = generated_text;
    return result;
}

} // namespace llama_wrapper

