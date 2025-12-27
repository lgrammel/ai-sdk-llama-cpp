import type {
  LanguageModelV1,
  LanguageModelV1CallOptions,
  LanguageModelV1CallWarning,
  LanguageModelV1FinishReason,
  LanguageModelV1StreamPart,
  LanguageModelV1FunctionToolCall,
  LanguageModelV1ProviderMetadata,
} from '@ai-sdk/provider';

import {
  loadModel,
  unloadModel,
  generate,
  generateStream,
  isModelLoaded,
  type LoadModelOptions,
  type GenerateOptions,
} from './native-binding.js';

export interface LlamaCppModelConfig {
  modelPath: string;
  contextSize?: number;
  gpuLayers?: number;
  threads?: number;
}

export interface LlamaCppGenerationConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
}

function convertFinishReason(reason: string): LanguageModelV1FinishReason {
  switch (reason) {
    case 'stop':
      return 'stop';
    case 'length':
      return 'length';
    default:
      return 'other';
  }
}

function formatPrompt(options: LanguageModelV1CallOptions): string {
  const { prompt, mode } = options;

  let formattedPrompt = '';

  // Handle system message
  if (mode.type === 'regular' || mode.type === 'object-json') {
    // System message handling
  }

  // Format messages into a prompt string
  for (const message of prompt) {
    switch (message.role) {
      case 'system':
        formattedPrompt += `<|system|>\n${message.content}\n`;
        break;
      case 'user':
        formattedPrompt += `<|user|>\n`;
        for (const part of message.content) {
          if (part.type === 'text') {
            formattedPrompt += part.text;
          }
          // Note: Image parts are not supported in this minimal implementation
        }
        formattedPrompt += '\n';
        break;
      case 'assistant':
        formattedPrompt += `<|assistant|>\n`;
        for (const part of message.content) {
          if (part.type === 'text') {
            formattedPrompt += part.text;
          }
        }
        formattedPrompt += '\n';
        break;
      case 'tool':
        // Tool results are not supported in this minimal implementation
        break;
    }
  }

  // Add assistant prefix for generation
  formattedPrompt += '<|assistant|>\n';

  return formattedPrompt;
}

export class LlamaCppLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = 'v1' as const;
  readonly provider = 'llama.cpp';
  readonly modelId: string;
  readonly defaultObjectGenerationMode = undefined;

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
      };

      this.modelHandle = await loadModel(options);
    })();

    await this.initPromise;
    this.initPromise = null;

    if (this.modelHandle === null) {
      throw new Error('Failed to load model');
    }

    return this.modelHandle;
  }

  async dispose(): Promise<void> {
    if (this.modelHandle !== null) {
      unloadModel(this.modelHandle);
      this.modelHandle = null;
    }
  }

  async doGenerate(options: LanguageModelV1CallOptions): Promise<{
    text: string;
    toolCalls?: LanguageModelV1FunctionToolCall[];
    finishReason: LanguageModelV1FinishReason;
    usage: { promptTokens: number; completionTokens: number };
    rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
    rawResponse?: { headers?: Record<string, string> };
    warnings?: LanguageModelV1CallWarning[];
    providerMetadata?: LanguageModelV1ProviderMetadata;
    logprobs?: undefined;
  }> {
    const handle = await this.ensureModelLoaded();

    const prompt = formatPrompt(options);

    const generateOptions: GenerateOptions = {
      prompt,
      maxTokens: options.maxTokens ?? 256,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      stopSequences: options.stopSequences,
    };

    const result = await generate(handle, generateOptions);

    return {
      text: result.text,
      finishReason: convertFinishReason(result.finishReason),
      usage: {
        promptTokens: result.promptTokens,
        completionTokens: result.completionTokens,
      },
      rawCall: {
        rawPrompt: prompt,
        rawSettings: generateOptions as unknown as Record<string, unknown>,
      },
    };
  }

  async doStream(options: LanguageModelV1CallOptions): Promise<{
    stream: ReadableStream<LanguageModelV1StreamPart>;
    rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
    rawResponse?: { headers?: Record<string, string> };
    warnings?: LanguageModelV1CallWarning[];
  }> {
    const handle = await this.ensureModelLoaded();

    const prompt = formatPrompt(options);

    const generateOptions: GenerateOptions = {
      prompt,
      maxTokens: options.maxTokens ?? 256,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      stopSequences: options.stopSequences,
    };

    let streamController: ReadableStreamDefaultController<LanguageModelV1StreamPart>;

    const stream = new ReadableStream<LanguageModelV1StreamPart>({
      start(controller) {
        streamController = controller;
      },
    });

    // Start generation in the background
    generateStream(handle, generateOptions, (token) => {
      streamController.enqueue({
        type: 'text-delta',
        textDelta: token,
      });
    })
      .then((result) => {
        streamController.enqueue({
          type: 'finish',
          finishReason: convertFinishReason(result.finishReason),
          usage: {
            promptTokens: result.promptTokens,
            completionTokens: result.completionTokens,
          },
        });
        streamController.close();
      })
      .catch((error) => {
        streamController.error(error);
      });

    return {
      stream,
      rawCall: {
        rawPrompt: prompt,
        rawSettings: generateOptions as unknown as Record<string, unknown>,
      },
    };
  }
}

