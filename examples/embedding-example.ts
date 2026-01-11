import { embedMany } from "ai";
import { llamaCpp } from "../dist/index.js";

const model = llamaCpp.embedding({
  modelPath: "./models/gemma-3-12b-it-Q3_K_M.gguf",
});

try {
  const result = await embedMany({
    model,
    values: ["sunny day at the beach", "rainy afternoon in the city"],
  });

  console.log("Embeddings:", result.embeddings.length);
  console.log("Dimensions:", result.embeddings[0].length);
  console.log();
  console.log("Usage:", result.usage);
} finally {
  await model.dispose();
}
