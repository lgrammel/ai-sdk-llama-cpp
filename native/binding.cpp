#include <napi.h>
#include "llama-wrapper.h"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>

// Global state for managing models
static std::unordered_map<int, std::unique_ptr<llama_wrapper::LlamaModel>> g_models;
static std::mutex g_models_mutex;
static std::atomic<int> g_next_handle{1};

// ============================================================================
// Async Workers
// ============================================================================

class LoadModelWorker : public Napi::AsyncWorker {
public:
    LoadModelWorker(
        Napi::Function& callback,
        const std::string& model_path,
        int n_gpu_layers,
        int n_ctx,
        int n_threads
    )
        : Napi::AsyncWorker(callback)
        , model_path_(model_path)
        , n_gpu_layers_(n_gpu_layers)
        , n_ctx_(n_ctx)
        , n_threads_(n_threads)
        , handle_(-1)
        , success_(false)
    {}

    void Execute() override {
        auto model = std::make_unique<llama_wrapper::LlamaModel>();

        llama_wrapper::ModelParams model_params;
        model_params.model_path = model_path_;
        model_params.n_gpu_layers = n_gpu_layers_;

        if (!model->load(model_params)) {
            SetError("Failed to load model from: " + model_path_);
            return;
        }

        llama_wrapper::ContextParams ctx_params;
        ctx_params.n_ctx = n_ctx_;
        ctx_params.n_threads = n_threads_;

        if (!model->create_context(ctx_params)) {
            SetError("Failed to create context");
            return;
        }

        handle_ = g_next_handle++;

        {
            std::lock_guard<std::mutex> lock(g_models_mutex);
            g_models[handle_] = std::move(model);
        }

        success_ = true;
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        Callback().Call({
            Env().Null(),
            Napi::Number::New(Env(), handle_)
        });
    }

    void OnError(const Napi::Error& e) override {
        Napi::HandleScope scope(Env());
        Callback().Call({
            Napi::String::New(Env(), e.Message()),
            Env().Null()
        });
    }

private:
    std::string model_path_;
    int n_gpu_layers_;
    int n_ctx_;
    int n_threads_;
    int handle_;
    bool success_;
};

class GenerateWorker : public Napi::AsyncWorker {
public:
    GenerateWorker(
        Napi::Function& callback,
        int handle,
        const std::string& prompt,
        const llama_wrapper::GenerationParams& params
    )
        : Napi::AsyncWorker(callback)
        , handle_(handle)
        , prompt_(prompt)
        , params_(params)
    {}

    void Execute() override {
        llama_wrapper::LlamaModel* model = nullptr;

        {
            std::lock_guard<std::mutex> lock(g_models_mutex);
            auto it = g_models.find(handle_);
            if (it == g_models.end()) {
                SetError("Invalid model handle");
                return;
            }
            model = it->second.get();
        }

        result_ = model->generate(prompt_, params_);
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());

        Napi::Object result = Napi::Object::New(Env());
        result.Set("text", Napi::String::New(Env(), result_.text));
        result.Set("promptTokens", Napi::Number::New(Env(), result_.prompt_tokens));
        result.Set("completionTokens", Napi::Number::New(Env(), result_.completion_tokens));
        result.Set("finishReason", Napi::String::New(Env(), result_.finish_reason));

        Callback().Call({Env().Null(), result});
    }

private:
    int handle_;
    std::string prompt_;
    llama_wrapper::GenerationParams params_;
    llama_wrapper::GenerationResult result_;
};

// Thread-safe function context for streaming
class StreamContext {
public:
    StreamContext(Napi::Env env, Napi::Function callback)
        : callback_(Napi::Persistent(callback))
        , env_(env)
    {}

    Napi::FunctionReference callback_;
    Napi::Env env_;
    llama_wrapper::GenerationResult result_;
};

void StreamCallJS(Napi::Env env, Napi::Function callback, StreamContext* context, const char* token) {
    if (env != nullptr && callback != nullptr) {
        if (token != nullptr) {
            // Streaming token
            callback.Call({
                env.Null(),
                Napi::String::New(env, "token"),
                Napi::String::New(env, token)
            });
        }
    }
}

class StreamGenerateWorker : public Napi::AsyncWorker {
public:
    StreamGenerateWorker(
        Napi::Function& callback,
        int handle,
        const std::string& prompt,
        const llama_wrapper::GenerationParams& params,
        Napi::Function& token_callback
    )
        : Napi::AsyncWorker(callback)
        , handle_(handle)
        , prompt_(prompt)
        , params_(params)
        , token_callback_(Napi::Persistent(token_callback))
    {}

    void Execute() override {
        llama_wrapper::LlamaModel* model = nullptr;

        {
            std::lock_guard<std::mutex> lock(g_models_mutex);
            auto it = g_models.find(handle_);
            if (it == g_models.end()) {
                SetError("Invalid model handle");
                return;
            }
            model = it->second.get();
        }

        // Collect tokens during generation
        result_ = model->generate_streaming(prompt_, params_, [this](const std::string& token) {
            std::lock_guard<std::mutex> lock(tokens_mutex_);
            tokens_.push_back(token);
            return true;
        });
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());

        // Call token callback for each collected token
        for (const auto& token : tokens_) {
            token_callback_.Call({
                Napi::String::New(Env(), token)
            });
        }

        // Final callback with result
        Napi::Object result = Napi::Object::New(Env());
        result.Set("text", Napi::String::New(Env(), result_.text));
        result.Set("promptTokens", Napi::Number::New(Env(), result_.prompt_tokens));
        result.Set("completionTokens", Napi::Number::New(Env(), result_.completion_tokens));
        result.Set("finishReason", Napi::String::New(Env(), result_.finish_reason));

        Callback().Call({Env().Null(), result});
    }

private:
    int handle_;
    std::string prompt_;
    llama_wrapper::GenerationParams params_;
    llama_wrapper::GenerationResult result_;
    Napi::FunctionReference token_callback_;
    std::vector<std::string> tokens_;
    std::mutex tokens_mutex_;
};

// ============================================================================
// N-API Functions
// ============================================================================

Napi::Value LoadModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Expected (options, callback)").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Object options = info[0].As<Napi::Object>();
    Napi::Function callback = info[1].As<Napi::Function>();

    std::string model_path = options.Get("modelPath").As<Napi::String>().Utf8Value();
    int n_gpu_layers = options.Has("gpuLayers") ?
        options.Get("gpuLayers").As<Napi::Number>().Int32Value() : 99;
    int n_ctx = options.Has("contextSize") ?
        options.Get("contextSize").As<Napi::Number>().Int32Value() : 2048;
    int n_threads = options.Has("threads") ?
        options.Get("threads").As<Napi::Number>().Int32Value() : 4;

    auto worker = new LoadModelWorker(callback, model_path, n_gpu_layers, n_ctx, n_threads);
    worker->Queue();

    return env.Undefined();
}

Napi::Value UnloadModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
        return env.Null();
    }

    int handle = info[0].As<Napi::Number>().Int32Value();

    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(handle);
        if (it != g_models.end()) {
            g_models.erase(it);
        }
    }

    return Napi::Boolean::New(env, true);
}

Napi::Value Generate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3 || !info[0].IsNumber() || !info[1].IsObject() || !info[2].IsFunction()) {
        Napi::TypeError::New(env, "Expected (handle, options, callback)").ThrowAsJavaScriptException();
        return env.Null();
    }

    int handle = info[0].As<Napi::Number>().Int32Value();
    Napi::Object options = info[1].As<Napi::Object>();
    Napi::Function callback = info[2].As<Napi::Function>();

    std::string prompt = options.Get("prompt").As<Napi::String>().Utf8Value();

    llama_wrapper::GenerationParams params;
    params.max_tokens = options.Has("maxTokens") ?
        options.Get("maxTokens").As<Napi::Number>().Int32Value() : 256;
    params.temperature = options.Has("temperature") ?
        options.Get("temperature").As<Napi::Number>().FloatValue() : 0.7f;
    params.top_p = options.Has("topP") ?
        options.Get("topP").As<Napi::Number>().FloatValue() : 0.9f;
    params.top_k = options.Has("topK") ?
        options.Get("topK").As<Napi::Number>().Int32Value() : 40;

    if (options.Has("stopSequences") && options.Get("stopSequences").IsArray()) {
        Napi::Array stop_arr = options.Get("stopSequences").As<Napi::Array>();
        for (uint32_t i = 0; i < stop_arr.Length(); i++) {
            params.stop_sequences.push_back(stop_arr.Get(i).As<Napi::String>().Utf8Value());
        }
    }

    auto worker = new GenerateWorker(callback, handle, prompt, params);
    worker->Queue();

    return env.Undefined();
}

Napi::Value GenerateStream(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 4 || !info[0].IsNumber() || !info[1].IsObject() ||
        !info[2].IsFunction() || !info[3].IsFunction()) {
        Napi::TypeError::New(env, "Expected (handle, options, tokenCallback, doneCallback)").ThrowAsJavaScriptException();
        return env.Null();
    }

    int handle = info[0].As<Napi::Number>().Int32Value();
    Napi::Object options = info[1].As<Napi::Object>();
    Napi::Function token_callback = info[2].As<Napi::Function>();
    Napi::Function done_callback = info[3].As<Napi::Function>();

    std::string prompt = options.Get("prompt").As<Napi::String>().Utf8Value();

    llama_wrapper::GenerationParams params;
    params.max_tokens = options.Has("maxTokens") ?
        options.Get("maxTokens").As<Napi::Number>().Int32Value() : 256;
    params.temperature = options.Has("temperature") ?
        options.Get("temperature").As<Napi::Number>().FloatValue() : 0.7f;
    params.top_p = options.Has("topP") ?
        options.Get("topP").As<Napi::Number>().FloatValue() : 0.9f;
    params.top_k = options.Has("topK") ?
        options.Get("topK").As<Napi::Number>().Int32Value() : 40;

    if (options.Has("stopSequences") && options.Get("stopSequences").IsArray()) {
        Napi::Array stop_arr = options.Get("stopSequences").As<Napi::Array>();
        for (uint32_t i = 0; i < stop_arr.Length(); i++) {
            params.stop_sequences.push_back(stop_arr.Get(i).As<Napi::String>().Utf8Value());
        }
    }

    auto worker = new StreamGenerateWorker(done_callback, handle, prompt, params, token_callback);
    worker->Queue();

    return env.Undefined();
}

Napi::Value IsModelLoaded(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "Expected model handle").ThrowAsJavaScriptException();
        return env.Null();
    }

    int handle = info[0].As<Napi::Number>().Int32Value();

    std::lock_guard<std::mutex> lock(g_models_mutex);
    auto it = g_models.find(handle);
    bool loaded = it != g_models.end() && it->second->is_loaded();

    return Napi::Boolean::New(env, loaded);
}

// ============================================================================
// Module Initialization
// ============================================================================

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("loadModel", Napi::Function::New(env, LoadModel));
    exports.Set("unloadModel", Napi::Function::New(env, UnloadModel));
    exports.Set("generate", Napi::Function::New(env, Generate));
    exports.Set("generateStream", Napi::Function::New(env, GenerateStream));
    exports.Set("isModelLoaded", Napi::Function::New(env, IsModelLoaded));
    return exports;
}

NODE_API_MODULE(llama_binding, Init)

