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

// Define a schema for a recipe using Zod
const RecipeSchema = z.object({
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
});

// Type inference from the schema
type Recipe = z.infer<typeof RecipeSchema>;

async function main() {
  // Create the model instance
  const model = llamaCpp({
    modelPath: "./models/gemma-3-12b-it-Q3_K_M.gguf",
    contextSize: 4096,
    // debug: true, // Uncomment to see verbose llama.cpp output
  });

  try {
    console.log("🍳 Generating a structured recipe...\n");

    // Generate a structured recipe object using AI SDK 6 syntax
    const { output: recipe } = await generateText({
      model,
      output: Output.object({
        schema: RecipeSchema,
      }),
      prompt:
        "Generate a simple recipe for toast with butter. Keep ingredient amounts short like '2 slices' or '1 tbsp'. Include 2 ingredients and 3 short steps.",
      maxTokens: 800,
    });

    if (!recipe) {
      throw new Error("No recipe was generated");
    }

    // The result is fully typed as Recipe
    console.log("📋 Generated Recipe:");
    console.log("=".repeat(50));
    console.log(`\n🍪 ${recipe.name}`);
    console.log(`\n📝 ${recipe.description}`);
    console.log(
      `\n⏱️  Prep: ${recipe.prepTime} min | Cook: ${recipe.cookTime} min`
    );
    console.log(`👥 Servings: ${recipe.servings}`);
    console.log(`📊 Difficulty: ${recipe.difficulty}`);

    console.log("\n🥣 Ingredients:");
    for (const ingredient of recipe.ingredients) {
      console.log(`   • ${ingredient.amount} ${ingredient.name}`);
    }

    console.log("\n📝 Steps:");
    recipe.steps.forEach((step, index) => {
      console.log(`   ${index + 1}. ${step}`);
    });

    console.log("\n" + "=".repeat(50));
    console.log("✅ Successfully generated structured recipe!\n");

    // Also demonstrate a simpler schema
    console.log("📊 Generating a simpler structured object...\n");

    const SimpleSchema = z.object({
      sentiment: z.enum(["positive", "negative", "neutral"]),
      score: z.number(),
      keywords: z.array(z.string()),
    });

    const { output: analysis } = await generateText({
      model,
      output: Output.object({
        schema: SimpleSchema,
      }),
      prompt:
        'Analyze the sentiment of: "I love this product! It works great and exceeded my expectations."',
      maxOutputTokens: 100,
    });

    if (!analysis) {
      throw new Error("No analysis was generated");
    }

    console.log("Sentiment Analysis Result:");
    console.log(`  Sentiment: ${analysis.sentiment}`);
    console.log(`  Score: ${analysis.score}`);
    console.log(`  Keywords: ${analysis.keywords.join(", ")}`);
    console.log();
  } catch (error) {
    console.error("Error during generation:", error);
  } finally {
    // Properly dispose the model to avoid memory leaks
    await model.dispose();
  }
}

main().catch(console.error);
