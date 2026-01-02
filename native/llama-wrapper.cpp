#include "llama-wrapper.h"
#include "llama.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace llama_wrapper {

// Global debug flag for log callback
static bool g_debug_mode = false;

// Custom log callback that respects debug mode
static void llama_log_callback(ggml_log_level level, const char *text, void *user_data) {
  (void)level;
  (void)user_data;
  if (g_debug_mode) {
    fprintf(stderr, "%s", text);
  }
}

//
// Batch utilities (adapted from llama.cpp/common/common.cpp)
//

// Reset batch token count for reuse
static void batch_clear(llama_batch &batch) {
  batch.n_tokens = 0;
}

// Add a single token to the batch with safety check
static void batch_add(llama_batch &batch, llama_token id, llama_pos pos, llama_seq_id seq_id,
                      bool logits) {
  assert(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

  batch.token[batch.n_tokens] = id;
  batch.pos[batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = 1;
  batch.seq_id[batch.n_tokens][0] = seq_id;
  batch.logits[batch.n_tokens] = logits;

  batch.n_tokens++;
}

// Add an entire sequence of tokens to the batch
static void batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens,
                          llama_seq_id seq_id, bool logits) {
  for (size_t i = 0; i < tokens.size(); i++) {
    batch_add(batch, tokens[i], static_cast<llama_pos>(i), seq_id, logits);
  }
}

LlamaModel::LlamaModel() = default;

LlamaModel::~LlamaModel() {
  unload();
}

LlamaModel::LlamaModel(LlamaModel &&other) noexcept
    : model_(other.model_), ctx_(other.ctx_), sampler_(other.sampler_),
      model_path_(std::move(other.model_path_)), chat_template_(std::move(other.chat_template_)) {
  other.model_ = nullptr;
  other.ctx_ = nullptr;
  other.sampler_ = nullptr;
}

LlamaModel &LlamaModel::operator=(LlamaModel &&other) noexcept {
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

bool LlamaModel::load(const ModelParams &params) {
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

bool LlamaModel::create_context(const ContextParams &params) {
  if (!model_) {
    return false;
  }

  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }

  llama_context_params ctx_params = llama_context_default_params();

  // Use model's training context size if not specified (0)
  if (params.n_ctx > 0) {
    ctx_params.n_ctx = params.n_ctx;
  } else {
    ctx_params.n_ctx = llama_model_n_ctx_train(model_);
  }

  ctx_params.n_batch = params.n_batch;
  ctx_params.n_threads = params.n_threads;
  ctx_params.n_threads_batch = params.n_threads;

  if (params.embedding) {
    ctx_params.embeddings = true;
    // UNSPECIFIED (-1) lets llama.cpp auto-detect from model metadata
    ctx_params.pooling_type = static_cast<enum llama_pooling_type>(params.pooling_type);
    // Utilize the full context
    if (ctx_params.n_batch < ctx_params.n_ctx) {
      ctx_params.n_batch = ctx_params.n_ctx;
    }
    // For encoder models (BERT, NomicBERT, etc.), n_ubatch must be >= n_tokens
    // for each decode call. Set n_ubatch = n_batch to support processing all
    // tokens at once. See: https://github.com/ggml-org/llama.cpp/issues/12836
    ctx_params.n_ubatch = ctx_params.n_batch;
  }

  ctx_ = llama_init_from_model(model_, ctx_params);
  return ctx_ != nullptr;
}

void LlamaModel::normalize_embedding(float *embedding, int n_embd) {
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

bool LlamaModel::is_encoder_model() const {
  if (!model_) {
    return false;
  }

  // Get the model architecture from GGUF metadata
  char arch[128];
  if (llama_model_meta_val_str(model_, "general.architecture", arch, sizeof(arch)) <= 0) {
    return false; // Can't determine, assume decoder
  }

  // Check 1: Explicit non-causal attention (e.g., BERT models)
  // Build the key: e.g., "nomic-bert.attention.causal"
  std::string causal_key = std::string(arch) + ".attention.causal";
  char causal_val[16];

  if (llama_model_meta_val_str(model_, causal_key.c_str(), causal_val, sizeof(causal_val)) > 0) {
    // "false" means non-causal attention (encoder model)
    if (strcmp(causal_val, "false") == 0) {
      return true;
    }
  }

  // Check 2: Explicit pooling type in GGUF metadata indicates an embedding model
  // (e.g., Qwen3-Embedding has pooling_type set but no attention.causal metadata)
  // Embedding models generally don't support multi-sequence batching well
  std::string pooling_key = std::string(arch) + ".pooling_type";
  char pooling_val[16];

  if (llama_model_meta_val_str(model_, pooling_key.c_str(), pooling_val, sizeof(pooling_val)) > 0) {
    // If pooling_type is explicitly set in model metadata, treat as embedding model
    // pooling_type > 0 means MEAN, CLS, LAST, or RANK pooling (not NONE)
    int pooling_type = atoi(pooling_val);
    if (pooling_type > 0) {
      return true;
    }
  }

  return false; // Default to decoder if metadata not found
}

std::vector<float> LlamaModel::embed_chunk(const std::vector<int32_t> &tokens, int seq_id,
                                           int n_embd, int pooling_type) {
  std::vector<float> embedding(n_embd, 0.0f);

  if (tokens.empty()) {
    return embedding;
  }

  // Clear the memory/KV cache
  llama_memory_t mem = llama_get_memory(ctx_);
  if (mem) {
    llama_memory_clear(mem, true);
  }

  // Create batch with sequence ID
  llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
  batch_add_seq(batch, tokens, seq_id, false); // Embeddings don't need logits

  // Decode to get embeddings
  if (llama_decode(ctx_, batch) != 0) {
    llama_batch_free(batch);
    throw std::runtime_error("llama_decode failed during embedding");
  }

  // Extract embedding based on pooling type
  const float *embd = nullptr;
  if (static_cast<enum llama_pooling_type>(pooling_type) == LLAMA_POOLING_TYPE_NONE) {
    // Get embedding for last token
    embd = llama_get_embeddings_ith(ctx_, tokens.size() - 1);
  } else {
    // Get pooled embedding for the sequence
    embd = llama_get_embeddings_seq(ctx_, seq_id);
  }

  if (embd) {
    std::copy(embd, embd + n_embd, embedding.begin());
  }

  llama_batch_free(batch);
  return embedding;
}

std::vector<std::vector<float>>
LlamaModel::embed_batch(const std::vector<std::vector<int32_t>> &all_tokens, int n_embd,
                        int pooling_type) {
  std::vector<std::vector<float>> embeddings;

  if (all_tokens.empty()) {
    return embeddings;
  }

  // Calculate total tokens needed
  size_t total_tokens = 0;
  for (const auto &tokens : all_tokens) {
    total_tokens += tokens.size();
  }

  // Clear the memory/KV cache
  llama_memory_t mem = llama_get_memory(ctx_);
  if (mem) {
    llama_memory_clear(mem, true);
  }

  // Create batch for all sequences
  llama_batch batch = llama_batch_init(total_tokens, 0, all_tokens.size());
  for (size_t seq_id = 0; seq_id < all_tokens.size(); seq_id++) {
    batch_add_seq(batch, all_tokens[seq_id], static_cast<llama_seq_id>(seq_id), false);
  }

  // Decode all sequences at once
  if (llama_decode(ctx_, batch) != 0) {
    llama_batch_free(batch);
    throw std::runtime_error("llama_decode failed during batch embedding");
  }

  // Extract embeddings for each sequence
  for (size_t seq_id = 0; seq_id < all_tokens.size(); seq_id++) {
    std::vector<float> embedding(n_embd, 0.0f);

    const float *embd = nullptr;
    if (static_cast<enum llama_pooling_type>(pooling_type) == LLAMA_POOLING_TYPE_NONE) {
      // For NONE pooling, we need to find the last token of this sequence
      // This is more complex in batched mode, fall back to sequence embedding
      embd = llama_get_embeddings_seq(ctx_, seq_id);
    } else {
      // Get pooled embedding for the sequence
      embd = llama_get_embeddings_seq(ctx_, seq_id);
    }

    if (embd) {
      std::copy(embd, embd + n_embd, embedding.begin());
    }

    embeddings.push_back(std::move(embedding));
  }

  llama_batch_free(batch);
  return embeddings;
}

EmbeddingResult LlamaModel::embed(const std::vector<std::string> &texts,
                                  const EmbedParams &params) {
  EmbeddingResult result;
  result.total_tokens = 0;

  if (!ctx_ || !model_) {
    return result;
  }

  // Verify embedding mode is enabled
  const enum llama_pooling_type pooling_type = llama_pooling_type(ctx_);
  if (pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
    throw std::runtime_error("Context not configured for embeddings");
  }

  const int n_embd = llama_model_n_embd(model_);
  const int n_ctx = llama_n_ctx(ctx_);
  // Clamp overlap to [0, n_ctx - 1] to prevent negative or zero step (infinite loop)
  const int raw_overlap = static_cast<int>(n_ctx * params.overlap);
  const int overlap = std::max(0, std::min(raw_overlap, n_ctx - 1));
  const int step = n_ctx - overlap;

  // Auto-detect BOS token preference from model metadata
  const llama_vocab *vocab = llama_model_get_vocab(model_);
  const bool add_bos = llama_vocab_get_add_bos(vocab);

  // First pass: tokenize all texts and check if batching is possible
  std::vector<std::vector<int32_t>> all_tokens;
  std::vector<size_t> long_text_indices; // Indices of texts that need chunking
  size_t total_short_tokens = 0;

  for (size_t i = 0; i < texts.size(); i++) {
    std::vector<int32_t> tokens = tokenize(texts[i], add_bos);
    result.total_tokens += tokens.size();

    if (static_cast<int>(tokens.size()) > n_ctx) {
      long_text_indices.push_back(i);
    } else if (!tokens.empty()) {
      total_short_tokens += tokens.size();
    }

    all_tokens.push_back(std::move(tokens));
  }

  // Determine if we can batch all short texts using multi-sequence batching.
  // Multi-sequence batching (different seq_ids for each text) only works reliably for
  // decoder (causal) models.
  bool can_batch = !is_encoder_model() && long_text_indices.empty() &&
                   total_short_tokens <= static_cast<size_t>(n_ctx);

  if (can_batch) {
    // Fast path: batch all short texts in a single decode call
    try {
      // Filter out empty texts for batching
      std::vector<std::vector<int32_t>> non_empty_tokens;
      std::vector<size_t> non_empty_indices;

      for (size_t i = 0; i < all_tokens.size(); i++) {
        if (!all_tokens[i].empty()) {
          non_empty_tokens.push_back(all_tokens[i]);
          non_empty_indices.push_back(i);
        }
      }

      std::vector<std::vector<float>> batch_embeddings;
      if (!non_empty_tokens.empty()) {
        batch_embeddings = embed_batch(non_empty_tokens, n_embd, pooling_type);
      }

      // Reconstruct results in original order
      size_t batch_idx = 0;
      for (size_t i = 0; i < all_tokens.size(); i++) {
        if (all_tokens[i].empty()) {
          result.embeddings.push_back(std::vector<float>(n_embd, 0.0f));
        } else {
          auto &emb = batch_embeddings[batch_idx++];
          if (params.normalize) {
            normalize_embedding(emb.data(), n_embd);
          }
          result.embeddings.push_back(std::move(emb));
        }
      }
    } catch (const std::runtime_error &e) {
      throw std::runtime_error(std::string("Failed to batch embed: ") + e.what());
    }
  } else {
    // Slow path: process each text individually (some need chunking or won't fit in batch)
    // Use seq_id=0 for all texts since we clear the cache between each text
    const int seq_id = 0;
    for (size_t text_idx = 0; text_idx < texts.size(); text_idx++) {
      const auto &tokens = all_tokens[text_idx];

      if (tokens.empty()) {
        result.embeddings.push_back(std::vector<float>(n_embd, 0.0f));
        continue;
      }

      try {
        if (static_cast<int>(tokens.size()) <= n_ctx) {
          // Process single chunk
          std::vector<float> embedding = embed_chunk(tokens, seq_id, n_embd, pooling_type);
          if (params.normalize) {
            normalize_embedding(embedding.data(), n_embd);
          }
          result.embeddings.push_back(std::move(embedding));
        } else {
          // Text exceeds context size - split into overlapping chunks
          std::vector<float> final_embedding(n_embd, 0.0f);
          int num_chunks = 0;

          for (size_t start = 0; start < tokens.size(); start += step) {
            size_t end = std::min(start + n_ctx, tokens.size());
            std::vector<int32_t> chunk_tokens(tokens.begin() + start, tokens.begin() + end);

            std::vector<float> chunk_emb = embed_chunk(chunk_tokens, seq_id, n_embd, pooling_type);
            for (int i = 0; i < n_embd; i++) {
              final_embedding[i] += chunk_emb[i];
            }
            num_chunks++;

            if (end == tokens.size()) {
              break;
            }
          }

          if (num_chunks > 0) {
            for (int i = 0; i < n_embd; i++) {
              final_embedding[i] /= static_cast<float>(num_chunks);
            }
          }

          if (params.normalize) {
            normalize_embedding(final_embedding.data(), n_embd);
          }
          result.embeddings.push_back(std::move(final_embedding));
        }
      } catch (const std::runtime_error &e) {
        throw std::runtime_error(std::string("Failed to embed text at index ") +
                                 std::to_string(text_idx) + ": " + e.what());
      }
    }
  }

  return result;
}

std::string LlamaModel::apply_chat_template(const std::vector<ChatMessage> &messages) {
  if (!model_) {
    return "";
  }

  // Determine which template to use
  const char *tmpl = nullptr;
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
  for (const auto &msg : messages) {
    llama_chat_message chat_msg;
    chat_msg.role = msg.role.c_str();
    chat_msg.content = msg.content.c_str();
    chat_messages.push_back(chat_msg);
  }

  // First call to get required buffer size
  int32_t result_size = llama_chat_apply_template(tmpl, chat_messages.data(), chat_messages.size(),
                                                  true, // add_ass: add assistant prompt
                                                  nullptr, 0);

  if (result_size < 0) {
    // Template not supported, return empty string
    return "";
  }

  // Allocate buffer and apply template
  std::vector<char> buffer(result_size + 1);
  llama_chat_apply_template(tmpl, chat_messages.data(), chat_messages.size(), true, buffer.data(),
                            buffer.size());

  return std::string(buffer.data(), result_size);
}

void LlamaModel::create_sampler(const GenerationParams &params) {
  if (sampler_) {
    llama_sampler_free(sampler_);
  }

  // Create a sampler chain
  sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());

  // Add grammar sampler first if grammar is provided (constrains token generation)
  if (!params.grammar.empty()) {
    const llama_vocab *vocab = llama_model_get_vocab(model_);
    llama_sampler *grammar_sampler =
        llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
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

std::vector<int32_t> LlamaModel::tokenize(const std::string &text, bool add_bos) {
  const llama_vocab *vocab = llama_model_get_vocab(model_);

  // First, get the number of tokens needed
  // When passing 0 for n_tokens_max, llama_tokenize returns negative of required size
  int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, add_bos, true);

  if (n_tokens < 0) {
    n_tokens = -n_tokens; // Convert to positive size
  }

  if (n_tokens == 0) {
    return {}; // Empty input
  }

  std::vector<int32_t> tokens(n_tokens);
  int actual_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(),
                                     tokens.size(), add_bos, true);

  if (actual_tokens < 0) {
    // Buffer still too small, resize and try again
    tokens.resize(-actual_tokens);
    actual_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(),
                                   add_bos, true);
  }

  if (actual_tokens > 0) {
    tokens.resize(actual_tokens);
  } else {
    tokens.clear();
  }

  return tokens;
}

std::string LlamaModel::detokenize(int32_t token) {
  const llama_vocab *vocab = llama_model_get_vocab(model_);

  char buf[256];
  int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
  if (n < 0) {
    return "";
  }
  return std::string(buf, n);
}

bool LlamaModel::is_eos_token(int32_t token) {
  const llama_vocab *vocab = llama_model_get_vocab(model_);
  return llama_vocab_is_eog(vocab, token);
}

GenerationResult LlamaModel::generate(const std::vector<ChatMessage> &messages,
                                      const GenerationParams &params) {
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
    for (const auto &stop_seq : params.stop_sequences) {
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
    if (should_stop)
      break;

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

GenerationResult LlamaModel::generate_streaming(const std::vector<ChatMessage> &messages,
                                                const GenerationParams &params,
                                                TokenCallback callback) {
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
    for (const auto &stop_seq : params.stop_sequences) {
      if (generated_text.length() >= stop_seq.length()) {
        if (generated_text.substr(generated_text.length() - stop_seq.length()) == stop_seq) {
          should_stop = true;
          result.finish_reason = "stop";
          break;
        }
      }
    }
    if (should_stop)
      break;

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
