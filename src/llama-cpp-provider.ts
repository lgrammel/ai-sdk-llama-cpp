import {
  LlamaCppLanguageModel,
  type LlamaCppModelConfig,
} from "./llama-cpp-language-model.js";
import {
  LlamaCppEmbeddingModel,
  type LlamaCppEmbeddingModelConfig,
} from "./llama-cpp-embedding-model.js";

export interface LlamaCppProviderConfig {
  /**
   * Path to the GGUF model file.
   */
  modelPath: string;

  /**
   * Maximum context size.
   * If not specified (or 0), uses the model's training context size from GGUF metadata.
   */
  contextSize?: number;

  /**
   * Number of layers to offload to GPU (default: 99, meaning all layers).
   * Set to 0 to disable GPU acceleration.
   */
  gpuLayers?: number;

  /**
   * Number of CPU threads to use (default: 4).
   */
  threads?: number;

  /**
   * Enable verbose debug output from llama.cpp (default: false).
   */
  debug?: boolean;
}

export interface LlamaCppProvider {
  (config: LlamaCppModelConfig): LlamaCppLanguageModel;
  languageModel(config: LlamaCppModelConfig): LlamaCppLanguageModel;
  embedding(config: LlamaCppEmbeddingModelConfig): LlamaCppEmbeddingModel;
}

function createLlamaCpp(): LlamaCppProvider {
  const provider = (config: LlamaCppModelConfig): LlamaCppLanguageModel => {
    const modelConfig: LlamaCppModelConfig = {
      modelPath: config.modelPath,
      contextSize: config.contextSize,
      gpuLayers: config.gpuLayers,
      threads: config.threads,
      debug: config.debug,
    };

    return new LlamaCppLanguageModel(modelConfig);
  };

  provider.languageModel = provider;

  provider.embedding = (config: LlamaCppEmbeddingModelConfig) => {
    return new LlamaCppEmbeddingModel(config);
  };

  return provider as LlamaCppProvider;
}

/**
 * Creates a llama.cpp model provider.
 *
 * @example
 * ```typescript
 * import { llamaCpp } from 'ai-sdk-llama-cpp';
 * import { embed, embedMany, generateText, streamText } from 'ai';
 *
 * const model = llamaCpp({
 *   modelPath: './models/llama-3.2-1b.gguf'
 * });
 *
 * const embeddingModel = llamaCpp.embedding({
 *   modelPath: './models/nomic-embed-text-v1.5.Q4_K_M.gguf'
 * });
 *
 * // Non-streaming
 * const { text } = await generateText({
 *   model,
 *   prompt: 'Hello, how are you?'
 * });
 *
 * // Streaming
 * const { textStream } = await streamText({
 *   model,
 *   prompt: 'Tell me a story'
 * });
 *
 * for await (const chunk of textStream) {
 *   process.stdout.write(chunk);
 * }
 *
 *
 * // Single embedding
 * const { embedding } = await embed({
 *   model: embeddingModel,
 *   value: 'Hello, world!'
 * });
 *
 * // Multiple embeddings
 * const { embeddings } = await embedMany({
 *   model: embeddingModel,
 *   values: ['Hello', 'World', 'How are you?']
 * });
 * ```
 */
export const llamaCpp = createLlamaCpp();

/**
 * Default export for convenience.
 */
export default llamaCpp;
