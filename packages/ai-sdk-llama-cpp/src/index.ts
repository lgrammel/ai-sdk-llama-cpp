export {
  llamaCpp,
  type LlamaCppProvider,
  type LlamaCppProviderConfig,
} from "./llama-cpp-provider.js";
export {
  LlamaCppLanguageModel,
  type LlamaCppModelConfig,
  type LlamaCppGenerationConfig,
  // Exported for testing
  convertMessages,
  convertFinishReason,
  convertUsage,
  // Tool calling exports
  generateToolCallGrammar,
  parseToolCalls,
  buildToolSystemPrompt,
  type ParsedToolCall,
} from "./llama-cpp-language-model.js";
export { LlamaCppEmbeddingModel } from "./llama-cpp-embedding-model.js";

// Export JSON schema to grammar converter for advanced use cases
export {
  convertJsonSchemaToGrammar,
  SchemaConverter,
  type SchemaConverterOptions,
} from "./json-schema-to-grammar.js";

// Re-export JSONSchema7 from @ai-sdk/provider for convenience
export type { JSONSchema7 } from "@ai-sdk/provider";

// Default export
export { default } from "./llama-cpp-provider.js";
