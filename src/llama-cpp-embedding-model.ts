import type {
  EmbeddingModelV3,
  EmbeddingModelV3CallOptions,
  EmbeddingModelV3Result,
  SharedV3Warning,
} from "@ai-sdk/provider";

import {
  loadModel,
  unloadModel,
  embed,
  isModelLoaded,
  type LoadModelOptions,
  type EmbedOptions,
} from "./native-binding.js";
import type { LlamaCppProviderConfig } from "./llama-cpp-provider.js";

export interface LlamaCppEmbeddingModelConfig extends LlamaCppProviderConfig {
  /**
   * Pooling type for embeddings (only applies to embedding models).
   * If not specified, auto-detects from the model's GGUF metadata.
   * - "none": Use last token embedding
   * - "mean": Average all token embeddings
   * - "cls": Use first token embedding (CLS token)
   * - "last": Use last token embedding
   */
  poolingType?: "none" | "mean" | "cls" | "last";

  /**
   * Whether to L2 normalize embeddings (default: true).
   * Set to true for cosine similarity, false for dot product.
   */
  normalize?: boolean;

  /**
   * Overlap fraction for chunking long texts (default: 0.1 = 10%).
   * Used when text exceeds the context size.
   */
  overlap?: number;
}

export class LlamaCppEmbeddingModel implements EmbeddingModelV3 {
  readonly specificationVersion = "v3" as const;
  readonly provider = "llama.cpp";
  readonly modelId: string;

  /**
   * Maximum number of embeddings that can be generated in a single call.
   * Local models can handle large batches, but we limit to prevent memory issues.
   */
  readonly maxEmbeddingsPerCall: number = 2048;

  /**
   * Whether the model supports parallel calls.
   * We use a single model instance, so parallel calls are not supported.
   */
  readonly supportsParallelCalls: boolean = false;

  private modelHandle: number | null = null;
  private readonly config: LlamaCppEmbeddingModelConfig;
  private initPromise: Promise<void> | null = null;

  constructor(config: LlamaCppEmbeddingModelConfig) {
    this.config = config;
    this.modelId = config.modelPath;
  }

  private getPoolingTypeNumber(): number {
    switch (this.config.poolingType) {
      case "none":
        return 0;
      case "mean":
        return 1;
      case "cls":
        return 2;
      case "last":
        return 3;
      default:
        return -1; // UNSPECIFIED: auto-detect from model metadata
    }
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
        contextSize: this.config.contextSize ?? 0,
        gpuLayers: this.config.gpuLayers ?? 99,
        threads: this.config.threads ?? 4,
        debug: this.config.debug ?? false,
        embedding: true,
        poolingType: this.getPoolingTypeNumber(),
      };

      this.modelHandle = await loadModel(options);
    })();

    await this.initPromise;
    this.initPromise = null;

    if (this.modelHandle === null) {
      throw new Error("Failed to load embedding model");
    }

    return this.modelHandle;
  }

  /**
   * Dispose of the model and free resources.
   */
  async dispose(): Promise<void> {
    // Wait for any pending initialization to complete
    if (this.initPromise) {
      await this.initPromise;
    }
    if (this.modelHandle !== null) {
      unloadModel(this.modelHandle);
      this.modelHandle = null;
    }
  }

  async doEmbed(
    options: EmbeddingModelV3CallOptions
  ): Promise<EmbeddingModelV3Result> {
    const handle = await this.ensureModelLoaded();

    const embedOptions: EmbedOptions = {
      texts: options.values,
      normalize: this.config.normalize ?? true,
      overlap: this.config.overlap ?? 0.1,
    };

    const result = await embed(handle, embedOptions);

    // Convert Float32Array[] to number[][]
    const embeddings: number[][] = result.embeddings.map((embedding) =>
      Array.from(embedding)
    );

    const warnings: SharedV3Warning[] = [];

    return {
      embeddings,
      usage: {
        tokens: result.totalTokens,
      },
      warnings,
    };
  }
}
