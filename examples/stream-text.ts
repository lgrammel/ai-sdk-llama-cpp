import { streamText } from "ai";
import { llamaCpp } from "../dist/index.js";

const model = llamaCpp({
  modelPath: "./models/Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
});

try {
  const result = streamText({
    model,
    prompt: "Invent a new holiday and describe its traditions.",
  });

  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }

  console.log();
  console.log();
  console.log("Usage:", await result.usage);
  console.log("Finish reason:", await result.finishReason);
} finally {
  await model.dispose();
}
