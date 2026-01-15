import { generateText, Output } from "ai";
import { z } from "zod";
import { llamaCpp } from "../dist/index.js";

const model = llamaCpp({ modelPath: "./models/gemma-3-12b-it-Q3_K_M.gguf" });

try {
  const recipeSchema = z.object({
    name: z.string(),
    ingredients: z.array(
      z.object({
        name: z.string(),
        amount: z.string(),
      })
    ),
    steps: z.array(z.string()),
  });

  const result = await generateText({
    model,
    prompt: "Generate a lasagna recipe.",
    output: Output.object({ schema: recipeSchema }),
  });

  console.log(JSON.stringify(result.output, null, 2));
  console.log();
  console.log("Usage:", result.usage);
  console.log("Finish reason:", result.finishReason);
} finally {
  await model.dispose();
}
