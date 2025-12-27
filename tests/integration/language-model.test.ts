import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { LanguageModelV3Message } from "@ai-sdk/provider";

// Mock the native binding module before importing the language model
vi.mock("../../src/native-binding.js", () => ({
  loadModel: vi.fn().mockResolvedValue(1),
  unloadModel: vi.fn().mockReturnValue(true),
  generate: vi.fn().mockResolvedValue({
    text: "Mock response text",
    promptTokens: 50,
    completionTokens: 10,
    finishReason: "stop",
  }),
  generateStream: vi.fn((handle, opts, onToken) => {
    // Simulate streaming tokens
    onToken("Hello");
    onToken(" ");
    onToken("world");
    onToken("!");
    return Promise.resolve({
      text: "Hello world!",
      promptTokens: 30,
      completionTokens: 4,
      finishReason: "stop",
    });
  }),
  isModelLoaded: vi.fn().mockReturnValue(true),
}));

// Import after mocking
import { LlamaCppLanguageModel } from "../../src/llama-cpp-language-model.js";
import * as nativeBinding from "../../src/native-binding.js";

describe("LlamaCppLanguageModel Integration", () => {
  let model: LlamaCppLanguageModel;

  beforeEach(() => {
    vi.clearAllMocks();
    model = new LlamaCppLanguageModel({
      modelPath: "/test/model.gguf",
      contextSize: 2048,
      gpuLayers: 99,
      threads: 4,
    });
  });

  afterEach(async () => {
    await model.dispose();
  });

  describe("doGenerate", () => {
    const testMessages: LanguageModelV3Message[] = [
      {
        role: "user",
        content: [{ type: "text", text: "Hello, how are you?" }],
      },
    ];

    it("returns valid LanguageModelV3GenerateResult structure", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
        maxOutputTokens: 100,
      });

      expect(result).toHaveProperty("content");
      expect(result).toHaveProperty("finishReason");
      expect(result).toHaveProperty("usage");
      expect(result).toHaveProperty("warnings");
      expect(result).toHaveProperty("request");
    });

    it("returns text content from generation", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe("text");
      expect(result.content[0]).toHaveProperty("text", "Mock response text");
    });

    it("returns correct finish reason", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.finishReason.unified).toBe("stop");
      expect(result.finishReason.raw).toBe("stop");
    });

    it("returns correct usage statistics", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.usage.inputTokens.total).toBe(50);
      expect(result.usage.outputTokens.total).toBe(10);
    });

    it("returns empty warnings array", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.warnings).toEqual([]);
    });

    it("includes request body in result", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
        maxOutputTokens: 256,
      });

      expect(result.request).toHaveProperty("body");
      expect(result.request.body).toHaveProperty("prompt");
      expect(result.request.body).toHaveProperty("maxTokens", 256);
    });

    it("applies default maxTokens when not specified", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.request.body).toHaveProperty("maxTokens", 256);
    });

    it("applies default temperature when not specified", async () => {
      const result = await model.doGenerate({
        prompt: testMessages,
      });

      expect(result.request.body).toHaveProperty("temperature", 0.7);
    });

    it("passes custom generation options", async () => {
      await model.doGenerate({
        prompt: testMessages,
        maxOutputTokens: 500,
        temperature: 0.5,
        topP: 0.8,
        topK: 30,
        stopSequences: ["END"],
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          maxTokens: 500,
          temperature: 0.5,
          topP: 0.8,
          topK: 30,
          stopSequences: ["END"],
        })
      );
    });

    it("formats prompt correctly for native binding", async () => {
      await model.doGenerate({
        prompt: testMessages,
      });

      expect(nativeBinding.generate).toHaveBeenCalledWith(
        expect.any(Number),
        expect.objectContaining({
          prompt: expect.stringContaining("<start_of_turn>user"),
        })
      );
    });
  });

  describe("doStream", () => {
    const testMessages: LanguageModelV3Message[] = [
      {
        role: "user",
        content: [{ type: "text", text: "Count to 3" }],
      },
    ];

    it("returns a ReadableStream", async () => {
      const result = await model.doStream({
        prompt: testMessages,
      });

      expect(result).toHaveProperty("stream");
      expect(result.stream).toBeInstanceOf(ReadableStream);
    });

    it("includes request body in result", async () => {
      const result = await model.doStream({
        prompt: testMessages,
        maxOutputTokens: 100,
      });

      expect(result.request).toHaveProperty("body");
      expect(result.request.body).toHaveProperty("maxTokens", 100);
    });

    it("emits stream-start as first part", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const reader = stream.getReader();
      const { value: firstPart } = await reader.read();

      expect(firstPart?.type).toBe("stream-start");
      expect(firstPart).toHaveProperty("warnings", []);

      reader.releaseLock();
    });

    it("emits text-start after stream-start", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);

      expect(parts[0].type).toBe("stream-start");
      expect(parts[1].type).toBe("text-start");
      expect(parts[1]).toHaveProperty("id");
    });

    it("emits text-delta parts for each token", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const textDeltas = parts.filter((p) => p.type === "text-delta");

      expect(textDeltas.length).toBe(4); // "Hello", " ", "world", "!"
      expect(textDeltas[0]).toHaveProperty("delta", "Hello");
      expect(textDeltas[1]).toHaveProperty("delta", " ");
      expect(textDeltas[2]).toHaveProperty("delta", "world");
      expect(textDeltas[3]).toHaveProperty("delta", "!");
    });

    it("emits text-end after all text-deltas", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const textEndIdx = parts.findIndex((p) => p.type === "text-end");
      const lastTextDeltaIdx = parts.reduce(
        (acc, p, i) => (p.type === "text-delta" ? i : acc),
        -1
      );

      expect(textEndIdx).toBeGreaterThan(lastTextDeltaIdx);
    });

    it("emits finish as last part", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const lastPart = parts[parts.length - 1];

      expect(lastPart.type).toBe("finish");
      expect(lastPart).toHaveProperty("finishReason");
      expect(lastPart).toHaveProperty("usage");
    });

    it("finish part contains correct finish reason", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const finishPart = parts.find((p) => p.type === "finish");

      expect(finishPart?.finishReason?.unified).toBe("stop");
    });

    it("finish part contains correct usage", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const finishPart = parts.find((p) => p.type === "finish");

      expect(finishPart?.usage?.inputTokens.total).toBe(30);
      expect(finishPart?.usage?.outputTokens.total).toBe(4);
    });

    it("all text-delta parts share the same id", async () => {
      const { stream } = await model.doStream({
        prompt: testMessages,
      });

      const parts = await collectStreamParts(stream);
      const textStart = parts.find((p) => p.type === "text-start");
      const textDeltas = parts.filter((p) => p.type === "text-delta");
      const textEnd = parts.find((p) => p.type === "text-end");

      const expectedId = textStart?.id;
      expect(expectedId).toBeDefined();

      for (const delta of textDeltas) {
        expect(delta.id).toBe(expectedId);
      }
      expect(textEnd?.id).toBe(expectedId);
    });
  });

  describe("model loading", () => {
    it("loads model lazily on first generation", async () => {
      const freshModel = new LlamaCppLanguageModel({
        modelPath: "/test/lazy.gguf",
      });

      expect(nativeBinding.loadModel).not.toHaveBeenCalled();

      await freshModel.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      expect(nativeBinding.loadModel).toHaveBeenCalledTimes(1);

      await freshModel.dispose();
    });

    it("reuses loaded model on subsequent calls", async () => {
      const messages: LanguageModelV3Message[] = [
        { role: "user", content: [{ type: "text", text: "test" }] },
      ];

      await model.doGenerate({ prompt: messages });
      await model.doGenerate({ prompt: messages });
      await model.doGenerate({ prompt: messages });

      expect(nativeBinding.loadModel).toHaveBeenCalledTimes(1);
    });

    it("passes correct options to loadModel", async () => {
      const customModel = new LlamaCppLanguageModel({
        modelPath: "/custom/path.gguf",
        contextSize: 4096,
        gpuLayers: 32,
        threads: 8,
        debug: true,
      });

      await customModel.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      expect(nativeBinding.loadModel).toHaveBeenCalledWith({
        modelPath: "/custom/path.gguf",
        contextSize: 4096,
        gpuLayers: 32,
        threads: 8,
        debug: true,
      });

      await customModel.dispose();
    });

    it("uses default values for optional config", async () => {
      const minimalModel = new LlamaCppLanguageModel({
        modelPath: "/minimal.gguf",
      });

      await minimalModel.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      expect(nativeBinding.loadModel).toHaveBeenCalledWith({
        modelPath: "/minimal.gguf",
        contextSize: 2048,
        gpuLayers: 99,
        threads: 4,
        debug: false,
      });

      await minimalModel.dispose();
    });
  });

  describe("dispose", () => {
    it("calls unloadModel with handle", async () => {
      // First generate to load the model
      await model.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      await model.dispose();

      expect(nativeBinding.unloadModel).toHaveBeenCalledWith(1);
    });

    it("can be called multiple times safely", async () => {
      await model.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      await model.dispose();
      await model.dispose();
      await model.dispose();

      // Should only unload once since modelHandle is null after first dispose
      expect(nativeBinding.unloadModel).toHaveBeenCalledTimes(1);
    });

    it("does nothing if model was never loaded", async () => {
      const unusedModel = new LlamaCppLanguageModel({
        modelPath: "/unused.gguf",
      });

      await unusedModel.dispose();

      expect(nativeBinding.unloadModel).not.toHaveBeenCalled();
    });
  });

  describe("error handling", () => {
    it("propagates generation errors", async () => {
      vi.mocked(nativeBinding.generate).mockRejectedValueOnce(
        new Error("Generation failed")
      );

      await expect(
        model.doGenerate({
          prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
        })
      ).rejects.toThrow("Generation failed");
    });

    it("propagates model loading errors", async () => {
      vi.mocked(nativeBinding.loadModel).mockRejectedValueOnce(
        new Error("Model not found")
      );

      const badModel = new LlamaCppLanguageModel({
        modelPath: "/nonexistent.gguf",
      });

      await expect(
        badModel.doGenerate({
          prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
        })
      ).rejects.toThrow("Model not found");
    });

    it("emits error part in stream on failure", async () => {
      vi.mocked(nativeBinding.generateStream).mockImplementationOnce(() => {
        throw new Error("Stream generation failed");
      });

      const { stream } = await model.doStream({
        prompt: [{ role: "user", content: [{ type: "text", text: "test" }] }],
      });

      const parts = await collectStreamParts(stream);
      const errorPart = parts.find((p) => p.type === "error");

      expect(errorPart).toBeDefined();
      expect(errorPart?.error).toBeInstanceOf(Error);
    });
  });
});

// Helper function to collect all stream parts
async function collectStreamParts(stream: ReadableStream<any>): Promise<any[]> {
  const parts: any[] = [];
  const reader = stream.getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
  }

  return parts;
}
