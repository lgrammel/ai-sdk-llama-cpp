import { generateText, stepCountIs, tool } from "ai";
import { z } from "zod";
import { llamaCpp } from "ai-sdk-llama-cpp";

/**
 * Example demonstrating tool calling with llama.cpp.
 *
 * This example shows how to use function tools with local LLMs.
 * The model will generate tool calls when appropriate, and you can
 * process the results and continue the conversation.
 */

// Fake weather data for demonstration
const weatherData: Record<string, { temperature: number; condition: string }> =
  {
    "new york": { temperature: 72, condition: "sunny" },
    "los angeles": { temperature: 85, condition: "clear" },
    london: { temperature: 59, condition: "cloudy" },
    tokyo: { temperature: 68, condition: "partly cloudy" },
    paris: { temperature: 64, condition: "rainy" },
    sydney: { temperature: 77, condition: "sunny" },
  };

// Create the model
const model = llamaCpp({
  modelPath: "../../models/gemma-3-12b-it-Q3_K_M.gguf",
  contextSize: 4096,
});

try {
  console.log("Calling model with weather tool...\n");

  // First call - the model should generate a tool call
  const result = await generateText({
    model,
    prompt: "What's the weather like in Tokyo?",
    maxOutputTokens: 500,
    tools: {
      getWeather: tool({
        description:
          "Get the current weather for a city. Returns temperature in Fahrenheit and weather condition.",
        inputSchema: z.object({
          city: z
            .string()
            .describe("The city name to get weather for (e.g., 'New York')"),
        }),
        execute: async ({ city }) => {
          // Simulate API call with fake data
          const normalizedCity = city.toLowerCase();
          const weather = weatherData[normalizedCity];

          if (weather) {
            return {
              city,
              temperature: weather.temperature,
              unit: "fahrenheit",
              condition: weather.condition,
            };
          } else {
            return {
              city,
              error: `Weather data not available for ${city}`,
            };
          }
        },
      }),
    },
    stopWhen: stepCountIs(10),
  });

  console.log("Final response:", result.text);
  console.log("\nUsage:", result.usage);
  console.log("Finish reason:", result.finishReason);

  // Log tool calls if any
  if (result.steps && result.steps.length > 0) {
    console.log("\nSteps:");
    for (const [index, step] of result.steps.entries()) {
      console.log(`  Step ${index + 1}:`);
      console.log(`    Finish reason: ${step.finishReason}`);
      if (step.toolCalls && step.toolCalls.length > 0) {
        console.log("    Tool calls:");
        for (const toolCall of step.toolCalls) {
          console.log(
            `      - ${toolCall.toolName}(${JSON.stringify(toolCall.input)})`
          );
        }
      }
      if (step.toolResults && step.toolResults.length > 0) {
        console.log("    Tool results:");
        for (const toolResult of step.toolResults) {
          console.log(
            `      - ${toolResult.toolName}: ${JSON.stringify(toolResult.output)}`
          );
        }
      }
    }
  }
} finally {
  await model.dispose();
}
