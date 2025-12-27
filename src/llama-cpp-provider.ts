import { LlamaCppLanguageModel, type LlamaCppModelConfig } from './llama-cpp-language-model.js';

export interface LlamaCppProviderConfig {
  /**
   * Path to the GGUF model file.
   */
  modelPath: string;

  /**
   * Maximum context size (default: 2048).
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

/**
 * Creates a llama.cpp language model provider.
 *
 * @example
 * ```typescript
 * import { createLlamaCpp } from 'ai-sdk-llama-cpp';
 * import { generateText, streamText } from 'ai';
 *
 * const model = createLlamaCpp({
 *   modelPath: './models/llama-3.2-1b.gguf'
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
 * ```
 */
export function createLlamaCpp(config: LlamaCppProviderConfig): LlamaCppLanguageModel {
  const modelConfig: LlamaCppModelConfig = {
    modelPath: config.modelPath,
    contextSize: config.contextSize,
    gpuLayers: config.gpuLayers,
    threads: config.threads,
    debug: config.debug,
  };

  return new LlamaCppLanguageModel(modelConfig);
}

/**
 * Default export for convenience.
 */
export default createLlamaCpp;

