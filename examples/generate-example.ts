import { generateText } from "ai";
import { createLlamaCpp } from "../dist/index.js";

async function main() {
  const model = createLlamaCpp({
    modelPath: "./models/gemma-3-12b-it-Q3_K_M.gguf",
    contextSize: 4096,
    // debug: true, // Uncomment to see verbose llama.cpp output
  });

  try {
    const { text } = await generateText({
      model,
      messages: [
        {
          role: "user",
          content: "Explain what a neural network is in 2-3 sentences.",
        },
      ],
      maxTokens: 150,
    });

    console.log(text);
  } catch (error) {
    console.error("Error during generation:", error);
  } finally {
    // Properly dispose the model to avoid Metal cleanup errors
    await model.dispose();
  }
}

main().catch(console.error);
