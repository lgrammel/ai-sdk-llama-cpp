# ai-sdk-llama-cpp

> **Alpha Software** - This package is in early development. The API may change between versions without notice.

> **macOS Only** - This package currently only supports macOS with Apple Silicon or Intel processors.

A minimal [llama.cpp](https://github.com/ggerganov/llama.cpp) provider for the [Vercel AI SDK](https://sdk.vercel.ai/), implementing the `LanguageModelV3` interface.

This package loads llama.cpp directly into Node.js memory via native C++ bindings, enabling local LLM inference without requiring an external server.

## Features

- **Native Performance**: Direct C++ bindings using node-addon-api (N-API)
- **GPU Acceleration**: Automatic Metal support on macOS
- **Streaming & Non-streaming**: Full support for both `generateText` and `streamText`
- **Structured Output**: Generate JSON objects with schema validation using `generateObject`
- **Chat Templates**: Automatic or configurable chat template formatting (llama3, chatml, gemma, etc.)
- **ESM Only**: Modern ECMAScript modules, no CommonJS
- **GGUF Support**: Load any GGUF-format model

## Prerequisites

Before installing, ensure you have the following:

- **macOS** (Apple Silicon or Intel)
- **Node.js** >= 18.0.0
- **CMake** >= 3.15
- **Xcode Command Line Tools**

```bash
# Install Xcode Command Line Tools (includes Clang)
xcode-select --install

# Install CMake via Homebrew
brew install cmake
```

## Installation

```bash
npm install ai-sdk-llama-cpp
```

The installation will automatically:

1. Detect macOS and verify platform compatibility
2. Compile llama.cpp as a static library with Metal support
3. Build the native Node.js addon

> **Note**: Installation on Windows or Linux will fail with an error. Only macOS is supported.

## Usage

### Basic Example

```typescript
import { generateText } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: "./models/llama-3.2-1b-instruct.Q4_K_M.gguf",
});

try {
  const { text } = await generateText({
    model,
    prompt: "Explain quantum computing in simple terms.",
  });

  console.log(text);
} finally {
  await model.dispose();
}
```

### Streaming Example

```typescript
import { streamText } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: "./models/llama-3.2-1b-instruct.Q4_K_M.gguf",
});

try {
  const result = streamText({
    model,
    prompt: "Write a haiku about programming.",
  });

  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }
} finally {
  await model.dispose();
}
```

### Structured Output

Generate type-safe JSON objects that conform to a schema using `generateObject`:

```typescript
import { generateObject } from "ai";
import { z } from "zod";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: "./models/your-model.gguf",
});

try {
  const { object: recipe } = await generateObject({
    model,
    schema: z.object({
      name: z.string(),
      ingredients: z.array(
        z.object({
          name: z.string(),
          amount: z.string(),
        })
      ),
      steps: z.array(z.string()),
    }),
    prompt: "Generate a recipe for chocolate chip cookies.",
  });

  // recipe is fully typed as { name: string, ingredients: {...}[], steps: string[] }
  console.log(recipe.name);
  console.log(recipe.ingredients);
  console.log(recipe.steps);
} finally {
  await model.dispose();
}
```

The structured output feature uses GBNF grammar constraints to ensure the model generates valid JSON that conforms to your schema. This works with:

- **Primitive types**: `string`, `number`, `integer`, `boolean`, `null`
- **Objects**: With `properties`, `required`, and `additionalProperties`
- **Arrays**: With `items`, `minItems`, `maxItems`
- **Enums and constants**: `enum`, `const`
- **Composition**: `oneOf`, `anyOf`, `allOf`
- **String constraints**: `minLength`, `maxLength`, `pattern`
- **Number constraints**: `minimum`, `maximum` (for integers)
- **String formats**: `date`, `time`, `date-time`, `uuid`
- **References**: Local `$ref` to `$defs`/`definitions`

### Embedding Example

```typescript
import { embed, embedMany } from "ai";
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp.embedding({
  modelPath: "./models/nomic-embed-text-v1.5.Q4_K_M.gguf",
});

try {
  const { embedding } = await embed({
    model,
    value: "Hello, world!",
  });

  const { embeddings } = await embedMany({
    model,
    values: ["Hello, world!", "Hello, ▲!"],
  });
} finally {
  model.dispose();
}
```

### Configuration Options

```typescript
const model = llamaCpp({
  // Required: Path to the GGUF model file
  modelPath: "./models/your-model.gguf",

  // Optional: Maximum context size (default: 2048)
  contextSize: 4096,

  // Optional: Number of layers to offload to GPU
  // Default: 99 (all layers). Set to 0 to disable GPU.
  gpuLayers: 99,

  // Optional: Number of CPU threads (default: 4)
  threads: 8,

  // Optional: Enable verbose debug output from llama.cpp (default: false)
  debug: true,

  // Optional: Chat template to use for formatting messages
  // - "auto" (default): Use the template embedded in the GGUF model file
  // - Template name: Use a specific built-in template (e.g., "llama3", "chatml", "gemma")
  chatTemplate: "auto",
});
```

#### Chat Templates

The `chatTemplate` option controls how messages are formatted before being sent to the model. Available templates include:

- `chatml`, `llama2`, `llama2-sys`, `llama3`, `llama4`
- `mistral-v1`, `mistral-v3`, `mistral-v7`
- `phi3`, `phi4`, `gemma`, `falcon3`, `zephyr`
- `deepseek`, `deepseek2`, `deepseek3`, `command-r`
- And more (see llama.cpp documentation for the full list)

### Generation Parameters

The standard AI SDK generation parameters are supported:

```typescript
try {
  const { text } = await generateText({
    model,
    prompt: "Hello!",
    maxTokens: 256, // Maximum tokens to generate
    temperature: 0.7, // Sampling temperature (0-2)
    topP: 0.9, // Nucleus sampling threshold
    topK: 40, // Top-k sampling
    stopSequences: ["\n"], // Stop generation at these sequences
  });
} finally {
  await model.dispose();
}
```

## Model Downloads

You'll need to download GGUF-format models separately. Popular sources:

- [Hugging Face](https://huggingface.co/models?search=gguf) - Search for GGUF models
- [TheBloke's Models](https://huggingface.co/TheBloke) - Popular quantized models

Example download:

```bash
# Create models directory
mkdir -p models

# Download a model (example: Llama 3.2 1B)
wget -P models/ https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

## API Reference

### `llamaCpp(config)`

Creates a new llama.cpp language model instance.

**Parameters:**

- `config.modelPath` (string, required): Path to the GGUF model file
- `config.contextSize` (number, optional): Maximum context size. Default: 2048
- `config.gpuLayers` (number, optional): GPU layers to offload. Default: 99
- `config.threads` (number, optional): CPU threads. Default: 4
- `config.debug` (boolean, optional): Enable verbose llama.cpp output. Default: false
- `config.chatTemplate` (string, optional): Chat template to use for formatting messages. Default: "auto"

**Returns:** `LlamaCppLanguageModel` - A language model compatible with the Vercel AI SDK

### `LlamaCppLanguageModel`

Implements the `LanguageModelV3` interface from `@ai-sdk/provider`.

**Methods:**

- `doGenerate(options)`: Non-streaming text generation
- `doStream(options)`: Streaming text generation
- `dispose()`: Unload the model and free GPU/CPU resources. **Always call this when done** to prevent memory leaks, especially when loading multiple models

## Limitations

This is a minimal implementation with the following limitations:

- **macOS only**: Windows and Linux are not supported
- **No tool/function calling**: Tool calls are not supported
- **No image inputs**: Only text prompts are supported

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/lgrammel/ai-sdk-llama-cpp.git
cd ai-sdk-llama-cpp

# Initialize submodules
git submodule update --init --recursive

# Install dependencies
npm install

# Build the native addon and TypeScript
npm run build
```

### Scripts

- `npm run build` - Build everything (native + TypeScript)
- `npm run build:native` - Build only the native addon
- `npm run build:ts` - Build only TypeScript
- `npm run clean` - Remove build artifacts
- `npm run test` - Run tests in watch mode
- `npm run test:run` - Run all tests once
- `npm run test:unit` - Run unit tests
- `npm run test:integration` - Run integration tests
- `npm run test:e2e` - Run end-to-end tests
- `npm run test:coverage` - Run tests with coverage

## License

MIT

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The underlying inference engine
- [Vercel AI SDK](https://sdk.vercel.ai/) - The AI SDK framework
- [node-addon-api](https://github.com/nodejs/node-addon-api) - N-API C++ wrapper
