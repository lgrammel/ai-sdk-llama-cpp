import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { embed, embedMany } from "ai";
import { llamaCpp, type LlamaCppEmbeddingModel } from "../../src/index.js";

/**
 * E2E tests for the llama.cpp provider.
 *
 * These tests require a real GGUF model file. Set the TEST_EMBEDDING_PATH
 * environment variable to run these tests:
 *
 *   TEST_EMBEDDING_PATH=./models/your-model.gguf npm run test:e2e
 *
 * If TEST_EMBEDDING_PATH is not set, these tests will be skipped.
 */

const TEST_EMBEDDING_PATH = process.env.TEST_EMBEDDING_PATH;
const shouldRunTests = !!TEST_EMBEDDING_PATH;

const describeE2E = shouldRunTests ? describe : describe.skip;

describeE2E("E2E Embedding Tests", () => {
  let model: LlamaCppEmbeddingModel;

  beforeAll(() => {
    if (!TEST_EMBEDDING_PATH) {
      throw new Error("TEST_EMBEDDING_PATH environment variable not set");
    }

    model = llamaCpp.embedding({
      modelPath: TEST_EMBEDDING_PATH,
      contextSize: 2048,
      gpuLayers: 0, // Use CPU for CI compatibility
      threads: 4,
    });
  });

  afterAll(async () => {
    if (model) {
      await model.dispose();
    }
  });

  describe("embed", () => {
    it(
      "embeds text",
      async () => {
        const { embedding, usage } = await embed({
          model,
          value: "Hello, world!",
        });

        expect(embedding.length).toBeGreaterThan(0);
        expect(usage.tokens).toBeGreaterThan(0);
      },
      { timeout: 120000 }
    );

    it(
      "embeds multiple texts",
      async () => {
        const { embeddings, usage } = await embedMany({
          model,
          values: ["Hello, world!", "Hello, universe!"],
        });

        expect(embeddings.length).toBe(2);
        expect(usage.tokens).toBeGreaterThan(0);
      },
      { timeout: 120000 }
    );
  });

  describe("model lifecycle", () => {
    it(
      "can create multiple model instances",
      async () => {
        if (!TEST_EMBEDDING_PATH) return;

        const model2 = llamaCpp.embedding({
          modelPath: TEST_EMBEDDING_PATH,
          contextSize: 1024,
        });

        const { embedding } = await embed({
          model: model2,
          value: "Hello, world!",
        });

        expect(embedding.length).toBeGreaterThan(0);

        await model2.dispose();
      },
      { timeout: 120000 }
    );

    it(
      "handles dispose gracefully",
      async () => {
        if (!TEST_EMBEDDING_PATH) return;

        const tempModel = llamaCpp.embedding({
          modelPath: TEST_EMBEDDING_PATH,
        });

        // Embed to load the model
        await embed({
          model: tempModel,
          value: "Hello, world!",
        });

        // Dispose should not throw
        await expect(tempModel.dispose()).resolves.toBeUndefined();
      },
      { timeout: 120000 }
    );
  });
});

// Test that runs without a model to verify skip behavior
describe("E2E Test Configuration", () => {
  it("TEST_EMBEDDING_PATH environment variable info", () => {
    if (!TEST_EMBEDDING_PATH) {
      console.log(
        "\n📋 E2E tests skipped: Set TEST_EMBEDDING_PATH to run with a real model"
      );
      console.log(
        "   Example: TEST_EMBEDDING_PATH=./models/model.gguf npm run test:e2e\n"
      );
    } else {
      console.log(
        `\n✅ Running E2E tests with model: ${TEST_EMBEDDING_PATH}\n`
      );
    }
    expect(true).toBe(true);
  });
});
