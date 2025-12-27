import { streamText } from "ai";
import { createLlamaCpp } from "../dist/index.js";

async function main() {
  const model = createLlamaCpp({
    modelPath: "./models/gemma-3-12b-it-Q3_K_M.gguf",
    contextSize: 4096,
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
  } catch (error) {
    console.error("Error during generation:", error);
  }
}

main().catch(console.error);
