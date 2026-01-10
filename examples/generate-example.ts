import { generateText } from "ai";
import { llamaCpp } from "../dist/index.js";

const model = llamaCpp({ modelPath: "./models/gemma-3-12b-it-Q3_K_M.gguf" });

const result = await generateText({
  model,
  prompt: "Invent a new holiday and describe its traditions.",
});

console.log(result.text);
console.log();
console.log("Usage:", result.usage);
console.log("Finish reason:", result.finishReason);

await model.dispose();
