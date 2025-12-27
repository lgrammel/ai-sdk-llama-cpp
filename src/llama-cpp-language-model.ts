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
} from "./native-binding.js";

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

export function formatPrompt(messages: LanguageModelV3Message[]): string {
  let formattedPrompt = "";

  // Format messages into a prompt string using Gemma-style format
  // This format works for most instruction-tuned models
  for (const message of messages) {
    switch (message.role) {
      case "system":
        // System messages go at the start
        formattedPrompt += `<start_of_turn>user\n${message.content}<end_of_turn>\n`;
        break;
      case "user":
        formattedPrompt += `<start_of_turn>user\n`;
        for (const part of message.content) {
          if (part.type === "text") {
            formattedPrompt += part.text;
          }
          // Note: File parts are not supported in this minimal implementation
        }
        formattedPrompt += `<end_of_turn>\n`;
        break;
      case "assistant":
        formattedPrompt += `<start_of_turn>model\n`;
        for (const part of message.content) {
          if (part.type === "text") {
            formattedPrompt += part.text;
          }
        }
        formattedPrompt += `<end_of_turn>\n`;
        break;
      case "tool":
        // Tool results are not supported in this minimal implementation
        break;
    }
  }

  // Add model prefix for generation
  formattedPrompt += "<start_of_turn>model\n";

  return formattedPrompt;
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

    const prompt = formatPrompt(options.prompt);

    const generateOptions: GenerateOptions = {
      prompt,
      maxTokens: options.maxOutputTokens ?? 256,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      stopSequences: options.stopSequences,
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

    const prompt = formatPrompt(options.prompt);

    const generateOptions: GenerateOptions = {
      prompt,
      maxTokens: options.maxOutputTokens ?? 256,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      stopSequences: options.stopSequences,
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
