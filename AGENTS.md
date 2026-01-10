# AGENTS.md

This file provides guidance for AI coding agents (Cursor, Copilot, Claude Code) working on this codebase.

## Project Overview

**ai-sdk-llama-cpp** is a llama.cpp provider for the Vercel AI SDK, implementing the `LanguageModelV3` interface. It loads llama.cpp directly into Node.js memory via native C++ bindings for local LLM inference.

**Platform Support**: macOS only (Apple Silicon or Intel)

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `npm install` |
| Build everything | `npm run build` |
| Build TypeScript only | `npm run build:ts` |
| Build native only | `npm run build:native` |
| Run all tests once | `npm run test:run` |
| Run unit tests | `npm run test:unit` |
| Run integration tests | `npm run test:integration` |
| Run E2E tests | `TEST_MODEL_PATH=./models/model.gguf npm run test:e2e` |
| Run example | `npx tsx examples/generate-example.ts` |
| Clean build artifacts | `npm run clean` |

## Setup & Installation

### Prerequisites

- **macOS** (Apple Silicon or Intel) - required
- **Node.js** >= 18.0.0
- **CMake** >= 3.15
- **Xcode Command Line Tools**

```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install CMake via Homebrew (if not already installed)
brew install cmake
```

### Installation Steps

```bash
# Clone and enter the repository
git clone https://github.com/lgrammel/ai-sdk-llama-cpp.git
cd ai-sdk-llama-cpp

# Initialize submodules (llama.cpp)
git submodule update --init --recursive

# Install dependencies (this also builds the native addon)
npm install

# Build TypeScript
npm run build:ts
```

The `npm install` step automatically:
1. Detects macOS and verifies platform compatibility
2. Compiles llama.cpp as a static library with Metal support
3. Builds the native Node.js addon

## Project Structure

```
├── src/                    # TypeScript source code
│   ├── index.ts            # Public exports
│   ├── llama-cpp-provider.ts    # Provider factory function
│   ├── llama-cpp-language-model.ts  # LanguageModelV3 implementation
│   ├── native-binding.ts   # Native module bindings
│   └── json-schema-to-grammar.ts   # JSON schema to GBNF grammar converter
├── native/                 # C++ native bindings
│   ├── binding.cpp         # N-API binding layer
│   ├── llama-wrapper.cpp   # llama.cpp wrapper implementation
│   └── llama-wrapper.h     # llama.cpp wrapper header
├── examples/               # Example usage files
├── tests/                  # Test suites
│   ├── unit/              # Unit tests (no model required)
│   ├── integration/       # Integration tests (mocked native bindings)
│   └── e2e/               # End-to-end tests (requires real model)
├── dist/                   # Compiled TypeScript output (generated)
└── build/                  # Native addon build output (generated)
```

## Testing

### Test Organization

- **Unit tests** (`tests/unit/`): Test pure functions and class instantiation. No model or native bindings required.
- **Integration tests** (`tests/integration/`): Test the language model class with mocked native bindings.
- **E2E tests** (`tests/e2e/`): Test actual inference with a real GGUF model file.

### Running Tests

```bash
# Run all tests once
npm run test:run

# Run tests in watch mode (for development)
npm run test

# Run specific test categories
npm run test:unit
npm run test:integration

# Run E2E tests (requires a GGUF model)
TEST_MODEL_PATH=./models/your-model.gguf npm run test:e2e

# Run tests with coverage
npm run test:coverage
```

### E2E Test Requirements

E2E tests require the `TEST_MODEL_PATH` environment variable to point to a valid GGUF model file. Without this, E2E tests are automatically skipped.

```bash
# Download a model for testing
mkdir -p models
wget -P models/ https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

# Run E2E tests
TEST_MODEL_PATH=./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf npm run test:e2e
```

### Writing Tests

- Use Vitest (`describe`, `it`, `expect`)
- Tests are in `tests/**/*.test.ts`
- Test timeout is 120 seconds (configured in `vitest.config.ts`)
- Use `describeE2E` helper from `tests/setup.ts` for conditional E2E tests

## Examples

### Running Examples

Examples require the TypeScript to be built first:

```bash
# Build TypeScript
npm run build:ts

# Run an example
npx tsx examples/generate-example.ts
npx tsx examples/stream-example.ts
npx tsx examples/structured-output-example.ts
```

### Example Structure

Examples follow this pattern:

```typescript
import { generateText } from "ai";
import { llamaCpp } from "../dist/index.js";

// Create model instance with config
const model = llamaCpp({ 
  modelPath: "./models/your-model.gguf",
  // Optional config: contextSize, gpuLayers, threads, debug, chatTemplate
});

try {
  // Use with AI SDK functions
  const result = await generateText({
    model,
    prompt: "Your prompt here",
  });

  console.log(result.text);
} finally {
  // Always dispose to free resources
  await model.dispose();
}
```

### Creating New Examples

1. Create a new file in `examples/` directory
2. Import from `"../dist/index.js"` (not `"../src/index.js"`)
3. Use `try/finally` to ensure `model.dispose()` is called
4. Update the model path to your local GGUF model
5. Run with `npx tsx examples/your-example.ts`

Example template:

```typescript
import { generateText, streamText, Output } from "ai";
import { z } from "zod";
import { llamaCpp } from "../dist/index.js";

const model = llamaCpp({ 
  modelPath: "./models/your-model.gguf",
  contextSize: 4096,  // optional
});

try {
  // Your example code here
  const { text } = await generateText({
    model,
    prompt: "Hello, world!",
    maxTokens: 100,
  });
  console.log(text);
} finally {
  await model.dispose();
}
```

## Code Style & Conventions

- **Module system**: ESM only (no CommonJS)
- **TypeScript**: Strict mode enabled
- **Target**: ES2022
- **Imports**: Use `.js` extensions for local imports (e.g., `import { foo } from "./bar.js"`)
- **Async/Await**: Preferred over raw Promises
- **Error handling**: Use try/finally for model lifecycle management

## Key APIs

### Provider Factory

```typescript
import { llamaCpp } from "ai-sdk-llama-cpp";

const model = llamaCpp({
  modelPath: string,        // Required: path to GGUF file
  contextSize?: number,     // Default: 2048
  gpuLayers?: number,       // Default: 99 (all layers)
  threads?: number,         // Default: 4
  debug?: boolean,          // Default: false
  chatTemplate?: string,    // Default: "auto"
});
```

### Model Methods

- `model.doGenerate(options)` - Non-streaming generation
- `model.doStream(options)` - Streaming generation
- `model.dispose()` - Free resources (always call when done)

### AI SDK Integration

Works with standard AI SDK functions:
- `generateText()` - Non-streaming text generation
- `streamText()` - Streaming text generation
- `generateObject()` - Structured output with schema
- `Output.object({ schema })` - For structured output mode

## Common Tasks

### Adding a New Feature

1. Implement in appropriate `src/` file
2. Export from `src/index.ts` if public API
3. Add unit tests in `tests/unit/`
4. Add integration tests in `tests/integration/`
5. Run `npm run test:run` to verify
6. Build with `npm run build:ts`

### Modifying Native Bindings

1. Edit files in `native/`
2. Rebuild with `npm run build:native`
3. Test with `npm run test:run`

### Debugging

- Enable verbose llama.cpp output: `llamaCpp({ modelPath, debug: true })`
- Run specific test: `npx vitest run tests/unit/provider.test.ts`
- Debug build: `npm run build:native:debug`

## Dependencies

- **Runtime**: `@ai-sdk/provider`, `cmake-js`, `node-addon-api`
- **Dev**: `ai`, `typescript`, `vitest`, `tsx`, `zod`

## Limitations

- macOS only (Windows/Linux not supported)
- No tool/function calling support
- No image input support (text only)
