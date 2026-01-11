import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3GenerateResult,
  LanguageModelV3Message,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  LanguageModelV3Usage,
  SharedV3Warning,
} from "@ai-sdk/provider";

import {
  loadModel,
  unloadModel,
  generate,
  generateStream,
  isModelLoaded,
  type LoadModelOptions,
  type GenerateOptions,
  type ChatMessage,
} from "./native-binding.js";

import type { JSONSchema7 } from "@ai-sdk/provider";
import { convertJsonSchemaToGrammar } from "./json-schema-to-grammar.js";

export interface LlamaCppModelConfig {
  modelPath: string;
  contextSize?: number;
  gpuLayers?: number;
  threads?: number;
  /**
   * Enable verbose debug output from llama.cpp.
   * Default: false
   */
  debug?: boolean;
  /**
   * Chat template to use for formatting messages.
   * - "auto" (default): Use the template embedded in the GGUF model file
   * - Template name: Use a specific built-in template (e.g., "llama3", "chatml", "gemma")
   *
   * Available templates: chatml, llama2, llama2-sys, llama3, llama4, mistral-v1,
   * mistral-v3, mistral-v7, phi3, phi4, gemma, falcon3, zephyr, deepseek, deepseek2,
   * deepseek3, command-r, and more.
   */
  chatTemplate?: string;
}

export interface LlamaCppGenerationConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
}

export function convertFinishReason(
  reason: string
): LanguageModelV3FinishReason {
  let unified: LanguageModelV3FinishReason["unified"];
  switch (reason) {
    case "stop":
      unified = "stop";
      break;
    case "length":
      unified = "length";
      break;
    default:
      unified = "other";
  }
  return { unified, raw: reason };
}

export function convertUsage(
  promptTokens: number,
  completionTokens: number
): LanguageModelV3Usage {
  return {
    inputTokens: {
      total: promptTokens,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: completionTokens,
      text: completionTokens,
      reasoning: undefined,
    },
  };
}

/**
 * Convert AI SDK messages to simple role/content format for the native layer.
 * The native layer will apply the appropriate chat template.
 */
export function convertMessages(messages: LanguageModelV3Message[]): ChatMessage[] {
  const result: ChatMessage[] = [];

  for (const message of messages) {
    switch (message.role) {
      case "system":
        result.push({
          role: "system",
          content: message.content,
        });
        break;
      case "user":
        // Extract text content from user messages
        let userContent = "";
        for (const part of message.content) {
          if (part.type === "text") {
            userContent += part.text;
          }
          // Note: File parts are not supported in this implementation
        }
        result.push({
          role: "user",
          content: userContent,
        });
        break;
      case "assistant":
        // Extract text content from assistant messages
        let assistantContent = "";
        for (const part of message.content) {
          if (part.type === "text") {
            assistantContent += part.text;
          }
        }
        result.push({
          role: "assistant",
          content: assistantContent,
        });
        break;
      case "tool":
        // Tool results are not supported in this implementation
        break;
    }
  }

  return result;
}

export class LlamaCppLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3" as const;
  readonly provider = "llama.cpp";
  readonly modelId: string;

  /**
   * Supported URL patterns - empty since we only support local files
   */
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private modelHandle: number | null = null;
  private readonly config: LlamaCppModelConfig;
  private initPromise: Promise<void> | null = null;

  constructor(config: LlamaCppModelConfig) {
    this.config = config;
    this.modelId = config.modelPath;
  }

  private async ensureModelLoaded(): Promise<number> {
    if (this.modelHandle !== null && isModelLoaded(this.modelHandle)) {
      return this.modelHandle;
    }

    if (this.initPromise) {
      await this.initPromise;
      if (this.modelHandle !== null) {
        return this.modelHandle;
      }
    }

    this.initPromise = (async () => {
      const options: LoadModelOptions = {
        modelPath: this.config.modelPath,
        contextSize: this.config.contextSize ?? 2048,
        gpuLayers: this.config.gpuLayers ?? 99,
        threads: this.config.threads ?? 4,
        debug: this.config.debug ?? false,
        chatTemplate: this.config.chatTemplate ?? "auto",
      };

      this.modelHandle = await loadModel(options);
    })();

    await this.initPromise;
    this.initPromise = null;

    if (this.modelHandle === null) {
      throw new Error("Failed to load model");
    }

    return this.modelHandle;
  }

  async dispose(): Promise<void> {
    if (this.modelHandle !== null) {
      unloadModel(this.modelHandle);
      this.modelHandle = null;
    }
  }

  async doGenerate(
    options: LanguageModelV3CallOptions
  ): Promise<LanguageModelV3GenerateResult> {
    const handle = await this.ensureModelLoaded();

    const messages = convertMessages(options.prompt);

    // Convert JSON schema to GBNF grammar if structured output is requested
    let grammar: string | undefined;
    if (options.responseFormat?.type === "json" && options.responseFormat.schema) {
      grammar = convertJsonSchemaToGrammar(options.responseFormat.schema as JSONSchema7);
    }

    const generateOptions: GenerateOptions = {
      messages,
      maxTokens: options.maxOutputTokens ?? 2048,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      stopSequences: options.stopSequences,
      grammar,
    };

    const result = await generate(handle, generateOptions);

    // Build content array with text content
    const content: LanguageModelV3Content[] = [
      {
        type: "text",
        text: result.text,
        providerMetadata: undefined,
      },
    ];

    const warnings: SharedV3Warning[] = [];

    return {
      content,
      finishReason: convertFinishReason(result.finishReason),
      usage: convertUsage(result.promptTokens, result.completionTokens),
      warnings,
      request: {
        body: generateOptions,
      },
    };
  }

  async doStream(
    options: LanguageModelV3CallOptions
  ): Promise<LanguageModelV3StreamResult> {
    const handle = await this.ensureModelLoaded();

    const messages = convertMessages(options.prompt);

    // Convert JSON schema to GBNF grammar if structured output is requested
    let grammar: string | undefined;
    if (options.responseFormat?.type === "json" && options.responseFormat.schema) {
      grammar = convertJsonSchemaToGrammar(options.responseFormat.schema as JSONSchema7);
    }

    const generateOptions: GenerateOptions = {
      messages,
      maxTokens: options.maxOutputTokens ?? 2048,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      stopSequences: options.stopSequences,
      grammar,
    };

    const textId = crypto.randomUUID();

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start: async (controller) => {
        try {
          // Emit stream start
          controller.enqueue({
            type: "stream-start",
            warnings: [],
          });

          // Emit text start
          controller.enqueue({
            type: "text-start",
            id: textId,
          });

          const result = await generateStream(
            handle,
            generateOptions,
            (token) => {
              controller.enqueue({
                type: "text-delta",
                id: textId,
                delta: token,
              });
            }
          );

          // Emit text end
          controller.enqueue({
            type: "text-end",
            id: textId,
          });

          // Emit finish
          controller.enqueue({
            type: "finish",
            finishReason: convertFinishReason(result.finishReason),
            usage: convertUsage(result.promptTokens, result.completionTokens),
          });

          controller.close();
        } catch (error) {
          controller.enqueue({
            type: "error",
            error,
          });
          controller.close();
        }
      },
    });

    return {
      stream,
      request: {
        body: generateOptions,
      },
    };
  }
}
