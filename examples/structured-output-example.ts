/**
 * Structured Output Example
 *
 * This example demonstrates how to use the llama.cpp provider with
 * the AI SDK's generateText function with Output.object() to produce
 * structured JSON output that conforms to a specific schema.
 *
 * Usage:
 *   npx tsx examples/structured-output-example.ts
 *
 * Make sure you have a GGUF model file in the models directory.
 */

import { generateText, Output } from "ai";
import { z } from "zod";
import { llamaCpp } from "../dist/index.js";

const model = llamaCpp({
  modelPath: "./models/gemma-3-12b-it-Q3_K_M.gguf",
  contextSize: 4096,
});

try {
  const { output: recipe } = await generateText({
    model,
    output: Output.object({
      schema: z.object({
        name: z.string().describe("The name of the recipe"),
        description: z.string().describe("A brief description of the dish"),
        prepTime: z.number().describe("Preparation time in minutes"),
        cookTime: z.number().describe("Cooking time in minutes"),
        servings: z.number().describe("Number of servings"),
        ingredients: z
          .array(
            z.object({
              name: z.string().describe("Ingredient name"),
              amount: z.string().describe("Amount with unit (e.g., '2 cups')"),
            })
          )
          .describe("List of ingredients"),
        steps: z.array(z.string()).describe("Cooking instructions"),
        difficulty: z
          .enum(["easy", "medium", "hard"])
          .describe("Difficulty level of the recipe"),
      }),
    }),
    prompt:
      "Generate a simple recipe for toast with butter. Keep ingredient amounts short like '2 slices' or '1 tbsp'. Include 2 ingredients and 3 short steps.",
    maxOutputTokens: 800,
  });

  console.dir(recipe, { depth: null });
} finally {
  await model.dispose();
}
