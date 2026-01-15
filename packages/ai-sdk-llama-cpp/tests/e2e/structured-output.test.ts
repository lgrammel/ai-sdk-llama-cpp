import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { generateText, streamText, Output } from "ai";
import { z } from "zod";
import { llamaCpp, LlamaCppLanguageModel } from "../../src/index.js";

/**
 * E2E tests for structured output with llama.cpp provider.
 *
 * These tests require a real GGUF model file. Set the TEST_MODEL_PATH
 * environment variable to run these tests:
 *
 *   TEST_MODEL_PATH=./models/your-model.gguf npm run test:e2e
 *
 * If TEST_MODEL_PATH is not set, these tests will be skipped.
 */

const TEST_MODEL_PATH = process.env.TEST_MODEL_PATH;
const shouldRunTests = !!TEST_MODEL_PATH;

const describeE2E = shouldRunTests ? describe : describe.skip;

describeE2E("E2E Structured Output Tests", () => {
  let model: LlamaCppLanguageModel;

  beforeAll(() => {
    if (!TEST_MODEL_PATH) {
      throw new Error("TEST_MODEL_PATH environment variable not set");
    }

    model = llamaCpp({
      modelPath: TEST_MODEL_PATH,
      contextSize: 4096,
      gpuLayers: 0, // Use CPU for CI compatibility
      threads: 4,
    });
  });

  afterAll(async () => {
    if (model) {
      await model.dispose();
    }
  });

  describe("generateText with Output.object()", () => {
    it(
      "generates a simple object with required fields",
      async () => {
        const schema = z.object({
          name: z.string(),
          age: z.number(),
        });

        const { output } = await generateText({
          model,
          output: Output.object({ schema }),
          prompt: "Generate a person named Alice who is 30 years old.",
          maxTokens: 100,
        });

        expect(output).toBeDefined();
        expect(typeof output!.name).toBe("string");
        expect(typeof output!.age).toBe("number");
      },
      { timeout: 120000 }
    );

    it(
      "generates an object with nested structure",
      async () => {
        const schema = z.object({
          title: z.string(),
          author: z.object({
            name: z.string(),
            email: z.string(),
          }),
        });

        const { output } = await generateText({
          model,
          output: Output.object({ schema }),
          prompt:
            'Generate a book titled "AI Basics" by John Smith with email john@example.com',
          maxTokens: 150,
        });

        expect(output).toBeDefined();
        expect(typeof output!.title).toBe("string");
        expect(typeof output!.author.name).toBe("string");
        expect(typeof output!.author.email).toBe("string");
      },
      { timeout: 120000 }
    );

    it(
      "generates an object with array fields",
      async () => {
        const schema = z.object({
          colors: z.array(z.string()),
        });

        const { output } = await generateText({
          model,
          output: Output.object({ schema }),
          prompt: "List 3 colors: red, blue, green.",
          maxTokens: 100,
        });

        expect(output).toBeDefined();
        expect(Array.isArray(output!.colors)).toBe(true);
        expect(output!.colors.length).toBeGreaterThan(0);
      },
      { timeout: 120000 }
    );

    it(
      "generates an object with enum constraint",
      async () => {
        const schema = z.object({
          status: z.enum(["pending", "active", "completed"]),
        });

        const { output } = await generateText({
          model,
          output: Output.object({ schema }),
          prompt: "Generate a status. Set it to active.",
          maxTokens: 50,
        });

        expect(output).toBeDefined();
        expect(["pending", "active", "completed"]).toContain(output!.status);
      },
      { timeout: 120000 }
    );

    it(
      "generates an object with boolean field",
      async () => {
        const schema = z.object({
          isEnabled: z.boolean(),
        });

        const { output } = await generateText({
          model,
          output: Output.object({ schema }),
          prompt: "Generate an object where isEnabled is true.",
          maxTokens: 50,
        });

        expect(output).toBeDefined();
        expect(typeof output!.isEnabled).toBe("boolean");
      },
      { timeout: 120000 }
    );

    it(
      "generates an object with optional fields",
      async () => {
        const schema = z.object({
          required_field: z.string(),
          optional_field: z.string().optional(),
        });

        const { output } = await generateText({
          model,
          output: Output.object({ schema }),
          prompt: 'Generate an object with required_field set to "hello".',
          maxTokens: 100,
        });

        expect(output).toBeDefined();
        expect(typeof output!.required_field).toBe("string");
      },
      { timeout: 120000 }
    );

    it(
      "generates a complex recipe object",
      async () => {
        const schema = z.object({
          name: z.string(),
          ingredients: z.array(
            z.object({
              name: z.string(),
              amount: z.string(),
            })
          ),
          steps: z.array(z.string()),
        });

        const { output } = await generateText({
          model,
          output: Output.object({ schema }),
          prompt:
            "Generate a simple recipe for toast with butter. Include 2 ingredients and 2 steps.",
          maxTokens: 300,
        });

        expect(output).toBeDefined();
        expect(typeof output!.name).toBe("string");
        expect(Array.isArray(output!.ingredients)).toBe(true);
        expect(Array.isArray(output!.steps)).toBe(true);

        if (output!.ingredients.length > 0) {
          expect(typeof output!.ingredients[0].name).toBe("string");
          expect(typeof output!.ingredients[0].amount).toBe("string");
        }
      },
      { timeout: 180000 }
    );
  });

  describe("streamText with Output.object()", () => {
    it(
      "streams a simple object",
      async () => {
        const schema = z.object({
          message: z.string(),
        });

        const { partialOutputStream } = streamText({
          model,
          output: Output.object({ schema }),
          prompt: 'Generate a greeting message saying "Hello World".',
          maxTokens: 50,
        });

        let lastOutput: z.infer<typeof schema> | undefined;
        for await (const partialOutput of partialOutputStream) {
          lastOutput = partialOutput as z.infer<typeof schema>;
        }

        expect(lastOutput).toBeDefined();
        expect(typeof lastOutput!.message).toBe("string");
      },
      { timeout: 120000 }
    );

    it(
      "provides usage after streaming object",
      async () => {
        const schema = z.object({
          value: z.number(),
        });

        const result = streamText({
          model,
          output: Output.object({ schema }),
          prompt: "Generate an object with value set to 42.",
          maxTokens: 50,
        });

        // Consume the stream
        for await (const _ of result.partialOutputStream) {
          // Just consume
        }

        const usage = await result.usage;
        expect(usage.promptTokens).toBeGreaterThan(0);
        expect(usage.completionTokens).toBeGreaterThan(0);
      },
      { timeout: 120000 }
    );
  });

  describe("edge cases", () => {
    it(
      "handles object with many fields",
      async () => {
        const schema = z.object({
          field1: z.string(),
          field2: z.string(),
          field3: z.string(),
          field4: z.number(),
          field5: z.boolean(),
        });

        const { output } = await generateText({
          model,
          output: Output.object({ schema }),
          prompt:
            'Generate an object with field1="a", field2="b", field3="c", field4=1, field5=true',
          maxTokens: 150,
        });

        expect(output).toBeDefined();
        expect(typeof output!.field1).toBe("string");
        expect(typeof output!.field2).toBe("string");
        expect(typeof output!.field3).toBe("string");
        expect(typeof output!.field4).toBe("number");
        expect(typeof output!.field5).toBe("boolean");
      },
      { timeout: 120000 }
    );

    it(
      "handles empty array in schema",
      async () => {
        const schema = z.object({
          items: z.array(z.string()),
        });

        const { output } = await generateText({
          model,
          output: Output.object({ schema }),
          prompt: "Generate an object with an empty items array.",
          maxTokens: 50,
        });

        expect(output).toBeDefined();
        expect(Array.isArray(output!.items)).toBe(true);
      },
      { timeout: 120000 }
    );
  });
});

// Test that runs without a model to verify skip behavior
describe("Structured Output Test Configuration", () => {
  it("TEST_MODEL_PATH environment variable info for structured output tests", () => {
    if (!TEST_MODEL_PATH) {
      console.log(
        "\n📋 Structured output E2E tests skipped: Set TEST_MODEL_PATH to run with a real model"
      );
      console.log(
        "   Example: TEST_MODEL_PATH=./models/model.gguf npm run test:e2e\n"
      );
    } else {
      console.log(
        `\n✅ Running structured output E2E tests with model: ${TEST_MODEL_PATH}\n`
      );
    }
    expect(true).toBe(true);
  });
});
