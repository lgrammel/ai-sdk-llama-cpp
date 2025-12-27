import { streamText } from "ai";
import { llamaCpp } from "../dist/index.js";

async function main() {
  const model = llamaCpp({
    modelPath: "./models/gemma-3-12b-it-Q3_K_M.gguf",
    contextSize: 4096,
    // debug: true, // Uncomment to see verbose llama.cpp output
  });

  try {
    const { textStream } = streamText({
      model,
      messages: [
        {
          role: "user",
          content:
            "Write a haiku about programming. Just the haiku, nothing else.",
        },
      ],
      maxOutputTokens: 100,
    });

    let hasOutput = false;
    for await (const chunk of textStream) {
      hasOutput = true;
      process.stdout.write(chunk);
    }

    if (!hasOutput) {
      console.log("(No output generated)");
    }

    console.log(); // Add newline after output
  } catch (error) {
    console.error("Error during generation:", error);
  } finally {
    // Properly dispose the model to avoid Metal cleanup errors
    await model.dispose();
  }
}

main().catch(console.error);
