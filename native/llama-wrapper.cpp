#include "llama-wrapper.h"
#include "llama.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <cmath>

namespace llama_wrapper {

// Global debug flag for log callback
static bool g_debug_mode = false;

// Custom log callback that respects debug mode
static void llama_log_callback(ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)user_data;
    if (g_debug_mode) {
        fprintf(stderr, "%s", text);
    }
}

LlamaModel::LlamaModel() = default;

LlamaModel::~LlamaModel() {
    unload();
}

LlamaModel::LlamaModel(LlamaModel&& other) noexcept
    : model_(other.model_)
    , ctx_(other.ctx_)
    , sampler_(other.sampler_)
    , model_path_(std::move(other.model_path_))
    , chat_template_(std::move(other.chat_template_)) {
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
        chat_template_ = std::move(other.chat_template_);
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

    // Set debug mode and install log callback
    g_debug_mode = params.debug;
    llama_log_set(llama_log_callback, nullptr);

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
    chat_template_ = params.chat_template;
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
    
    if (params.embedding) {
        ctx_params.embeddings = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    }

    ctx_ = llama_init_from_model(model_, ctx_params);
    return ctx_ != nullptr;
}

void LlamaModel::normalize_embedding(float* embedding, int n_embd) {
    float sum = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        sum += embedding[i] * embedding[i];
    }
    float norm = std::sqrt(sum);
    if (norm > 0.0f) {
        for (int i = 0; i < n_embd; i++) {
            embedding[i] /= norm;
        }
    }
}

EmbeddingResult LlamaModel::embed(const std::vector<std::string>& texts) {
    EmbeddingResult result;
    result.total_tokens = 0;

    if (!ctx_ || !model_) {
        return result;
    }

    const int n_embd = llama_model_n_embd(model_);
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx_);

    // Process each text
    for (size_t seq_id = 0; seq_id < texts.size(); seq_id++) {
        const std::string& text = texts[seq_id];

        // Tokenize the text
        std::vector<int32_t> tokens = tokenize(text, true);
        result.total_tokens += tokens.size();

        if (tokens.empty()) {
            // Return zero embedding for empty text
            result.embeddings.push_back(std::vector<float>(n_embd, 0.0f));
            continue;
        }

        // Clear the memory/KV cache
        llama_memory_t mem = llama_get_memory(ctx_);
        if (mem) {
            llama_memory_clear(mem, true);
        }

        // Create batch with sequence ID
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i] = true;  // We want embeddings for all tokens
        }
        batch.n_tokens = tokens.size();

        // Decode to get embeddings
        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            result.embeddings.push_back(std::vector<float>(n_embd, 0.0f));
            continue;
        }

        // Extract embedding based on pooling type
        const float* embd = nullptr;
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // Get embedding for last token
            embd = llama_get_embeddings_ith(ctx_, tokens.size() - 1);
        } else {
            // Get pooled embedding for the sequence
            embd = llama_get_embeddings_seq(ctx_, seq_id);
        }

        if (embd) {
            std::vector<float> embedding(n_embd);
            std::copy(embd, embd + n_embd, embedding.begin());
            // Normalize the embedding (L2 normalization)
            normalize_embedding(embedding.data(), n_embd);
            result.embeddings.push_back(std::move(embedding));
        } else {
            result.embeddings.push_back(std::vector<float>(n_embd, 0.0f));
        }

        llama_batch_free(batch);
    }

    return result;
}

std::string LlamaModel::apply_chat_template(const std::vector<ChatMessage>& messages) {
    if (!model_) {
        return "";
    }

    // Determine which template to use
    const char* tmpl = nullptr;
    if (chat_template_ == "auto") {
        // Use the template embedded in the model
        tmpl = llama_model_chat_template(model_, nullptr);
    } else {
        // Use the specified template name
        tmpl = chat_template_.c_str();
    }

    // Convert messages to llama_chat_message format
    std::vector<llama_chat_message> chat_messages;
    chat_messages.reserve(messages.size());
    for (const auto& msg : messages) {
        llama_chat_message chat_msg;
        chat_msg.role = msg.role.c_str();
        chat_msg.content = msg.content.c_str();
        chat_messages.push_back(chat_msg);
    }

    // First call to get required buffer size
    int32_t result_size = llama_chat_apply_template(
        tmpl,
        chat_messages.data(),
        chat_messages.size(),
        true,  // add_ass: add assistant prompt
        nullptr,
        0
    );

    if (result_size < 0) {
        // Template not supported, return empty string
        return "";
    }

    // Allocate buffer and apply template
    std::vector<char> buffer(result_size + 1);
    llama_chat_apply_template(
        tmpl,
        chat_messages.data(),
        chat_messages.size(),
        true,
        buffer.data(),
        buffer.size()
    );

    return std::string(buffer.data(), result_size);
}

void LlamaModel::create_sampler(const GenerationParams& params) {
    if (sampler_) {
        llama_sampler_free(sampler_);
    }

    // Create a sampler chain
    sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Add grammar sampler first if grammar is provided (constrains token generation)
    if (!params.grammar.empty()) {
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        llama_sampler* grammar_sampler = llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
        if (grammar_sampler) {
            llama_sampler_chain_add(sampler_, grammar_sampler);
        }
    }

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

GenerationResult LlamaModel::generate(const std::vector<ChatMessage>& messages, const GenerationParams& params) {
    GenerationResult result;
    result.finish_reason = "error";

    if (!ctx_ || !model_) {
        return result;
    }

    // Apply chat template to get the prompt
    std::string prompt = apply_chat_template(messages);
    if (prompt.empty()) {
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
    const std::vector<ChatMessage>& messages,
    const GenerationParams& params,
    TokenCallback callback
) {
    GenerationResult result;
    result.finish_reason = "error";

    if (!ctx_ || !model_) {
        return result;
    }

    // Apply chat template to get the prompt
    std::string prompt = apply_chat_template(messages);
    if (prompt.empty()) {
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

