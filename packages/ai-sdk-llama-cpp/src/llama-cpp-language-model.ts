import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3FunctionTool,
  LanguageModelV3GenerateResult,
  LanguageModelV3Message,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  LanguageModelV3Usage,
  SharedV3Warning,
} from "@ai-sdk/provider";

import {
  loadModel,
  unloadModel,
  generate,
  generateStream,
  isModelLoaded,
  type LoadModelOptions,
  type GenerateOptions,
  type ChatMessage,
} from "./native-binding.js";

import type { JSONSchema7 } from "@ai-sdk/provider";
import {
  convertJsonSchemaToGrammar,
  SchemaConverter,
} from "./json-schema-to-grammar.js";

export interface LlamaCppModelConfig {
  modelPath: string;
  contextSize?: number;
  gpuLayers?: number;
  threads?: number;
  /**
   * Enable verbose debug output from llama.cpp.
   * Default: false
   */
  debug?: boolean;
  /**
   * Chat template to use for formatting messages.
   * - "auto" (default): Use the template embedded in the GGUF model file
   * - Template name: Use a specific built-in template (e.g., "llama3", "chatml", "gemma")
   *
   * Available templates: chatml, llama2, llama2-sys, llama3, llama4, mistral-v1,
   * mistral-v3, mistral-v7, phi3, phi4, gemma, falcon3, zephyr, deepseek, deepseek2,
   * deepseek3, command-r, and more.
   */
  chatTemplate?: string;
}

export interface LlamaCppGenerationConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
}

export function convertFinishReason(
  reason: string
): LanguageModelV3FinishReason {
  let unified: LanguageModelV3FinishReason["unified"];
  switch (reason) {
    case "stop":
      unified = "stop";
      break;
    case "length":
      unified = "length";
      break;
    default:
      unified = "other";
  }
  return { unified, raw: reason };
}

export function convertUsage(
  promptTokens: number,
  completionTokens: number
): LanguageModelV3Usage {
  return {
    inputTokens: {
      total: promptTokens,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: completionTokens,
      text: completionTokens,
      reasoning: undefined,
    },
  };
}

/**
 * Represents a parsed tool call from the model output.
 */
export interface ParsedToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

/**
 * Generate a GBNF grammar for tool calls based on the provided tool definitions.
 * This grammar constrains the model to produce valid JSON tool calls.
 */
export function generateToolCallGrammar(
  tools: LanguageModelV3FunctionTool[]
): string {
  // Create a grammar that allows the model to output a tool call
  // Format: {"tool_calls":[{"id":"...","name":"...","arguments":{...}}]}

  // Generate the tool-specific argument schemas
  const toolGrammars: string[] = [];

  for (const tool of tools) {
    const converter = new SchemaConverter();
    converter.resolveRefs(tool.inputSchema as JSONSchema7);
    converter.visit(tool.inputSchema as JSONSchema7, `${tool.name}-args`);

    // Get the grammar rules for this tool's arguments
    const argGrammar = converter.formatGrammar();

    // Extract just the rules (without the root rule)
    const lines = argGrammar
      .split("\n")
      .filter((line) => line.trim() && !line.startsWith("root "));

    toolGrammars.push(...lines);
  }

  // Build the complete grammar
  const toolNameAlternatives = tools
    .map((t) => `"\\"${t.name}\\""`)
    .join(" | ");

  // Build arguments alternatives based on tool names
  const toolArgsAlternatives = tools.map((t) => `${t.name}-args`).join(" | ");

  // Combine all grammars
  const uniqueRules = new Map<string, string>();

  // Add tool-specific argument rules
  for (const line of toolGrammars) {
    const match = line.match(/^(\S+)\s*::=\s*(.+)$/);
    if (match) {
      uniqueRules.set(match[1], match[2]);
    }
  }

  // Build the final grammar
  let grammar = "";

  // Root rule - a tool call object
  grammar += `root ::= "{" space tool-calls-kv "}" space\n`;
  grammar += `tool-calls-kv ::= "\\"tool_calls\\"" space ":" space "[" space tool-call (space "," space tool-call)* space "]"\n`;
  grammar += `tool-call ::= "{" space id-kv "," space name-kv "," space args-kv "}" space\n`;
  grammar += `id-kv ::= "\\"id\\"" space ":" space string\n`;
  grammar += `name-kv ::= "\\"name\\"" space ":" space tool-name\n`;
  grammar += `tool-name ::= ${toolNameAlternatives}\n`;
  grammar += `args-kv ::= "\\"arguments\\"" space ":" space tool-args\n`;
  grammar += `tool-args ::= ${toolArgsAlternatives}\n`;

  // Add all the tool-specific rules
  for (const [name, rule] of uniqueRules) {
    grammar += `${name} ::= ${rule}\n`;
  }

  // Add common rules
  grammar += `space ::= | " " | "\\n"{1,2} [ \\t]{0,20}\n`;
  grammar += `string ::= "\\"" char* "\\"" space\n`;
  grammar += `char ::= [^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n`;

  return grammar;
}

/**
 * Parse the model output to extract tool calls.
 * Returns null if the output is not a valid tool call JSON.
 */
export function parseToolCalls(text: string): ParsedToolCall[] | null {
  try {
    // Try to parse as JSON
    const trimmed = text.trim();
    if (!trimmed.startsWith("{")) {
      return null;
    }

    const parsed = JSON.parse(trimmed);

    // Check if it has the tool_calls structure
    if (!parsed.tool_calls || !Array.isArray(parsed.tool_calls)) {
      return null;
    }

    const toolCalls: ParsedToolCall[] = [];

    for (const call of parsed.tool_calls) {
      if (
        typeof call.id === "string" &&
        typeof call.name === "string" &&
        typeof call.arguments === "object"
      ) {
        toolCalls.push({
          id: call.id,
          name: call.name,
          arguments: call.arguments,
        });
      }
    }

    return toolCalls.length > 0 ? toolCalls : null;
  } catch {
    return null;
  }
}

/**
 * Generate a unique tool call ID.
 */
function generateToolCallId(): string {
  return `call_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

/**
 * Build a system prompt that instructs the model to use tools.
 */
export function buildToolSystemPrompt(
  tools: LanguageModelV3FunctionTool[]
): string {
  const toolDescriptions = tools
    .map((tool) => {
      const params = JSON.stringify(tool.inputSchema, null, 2);
      return `- ${tool.name}: ${tool.description || "No description"}\n  Parameters: ${params}`;
    })
    .join("\n\n");

  return `You have access to the following tools:

${toolDescriptions}

When you need to use a tool, respond with a JSON object in this exact format:
{"tool_calls":[{"id":"call_<unique_id>","name":"<tool_name>","arguments":{<tool_arguments>}}]}

Important:
- The "id" should be a unique identifier starting with "call_"
- The "name" must exactly match one of the available tool names
- The "arguments" must match the tool's parameter schema
- Only respond with the JSON when you want to call a tool
- You can call multiple tools by adding more objects to the tool_calls array`;
}

/**
 * Convert AI SDK messages to simple role/content format for the native layer.
 * The native layer will apply the appropriate chat template.
 */
export function convertMessages(
  messages: LanguageModelV3Message[],
  tools?: LanguageModelV3FunctionTool[]
): ChatMessage[] {
  const result: ChatMessage[] = [];

  // Add tool system prompt if tools are provided
  if (tools && tools.length > 0) {
    result.push({
      role: "system",
      content: buildToolSystemPrompt(tools),
    });
  }

  for (const message of messages) {
    switch (message.role) {
      case "system":
        result.push({
          role: "system",
          content: message.content,
        });
        break;
      case "user":
        // Extract text content from user messages
        let userContent = "";
        for (const part of message.content) {
          if (part.type === "text") {
            userContent += part.text;
          }
          // Note: File parts are not supported in this implementation
        }
        result.push({
          role: "user",
          content: userContent,
        });
        break;
      case "assistant":
        // Extract text and tool call content from assistant messages
        let assistantContent = "";
        const toolCallParts: Array<{
          toolCallId: string;
          toolName: string;
          input: unknown;
        }> = [];

        for (const part of message.content) {
          if (part.type === "text") {
            assistantContent += part.text;
          } else if (part.type === "tool-call") {
            toolCallParts.push({
              toolCallId: part.toolCallId,
              toolName: part.toolName,
              input: part.input,
            });
          }
        }

        // If there are tool calls, format them as JSON
        if (toolCallParts.length > 0) {
          const toolCallsJson = JSON.stringify({
            tool_calls: toolCallParts.map((tc) => ({
              id: tc.toolCallId,
              name: tc.toolName,
              arguments: tc.input,
            })),
          });
          assistantContent = toolCallsJson;
        }

        if (assistantContent) {
          result.push({
            role: "assistant",
            content: assistantContent,
          });
        }
        break;
      case "tool":
        // Handle tool results - format them as user messages with the result
        for (const part of message.content) {
          if (part.type === "tool-result") {
            const output = part.output;
            let resultText = "";

            if (output.type === "text") {
              resultText = output.value;
            } else if (output.type === "json") {
              resultText = JSON.stringify(output.value);
            } else if (output.type === "error-text") {
              resultText = `Error: ${output.value}`;
            } else if (output.type === "error-json") {
              resultText = `Error: ${JSON.stringify(output.value)}`;
            } else if (output.type === "execution-denied") {
              resultText = `Execution denied${output.reason ? `: ${output.reason}` : ""}`;
            } else if (output.type === "content") {
              // Convert content array to text representation
              resultText = output.value
                .map((item) => {
                  if (item.type === "text") {
                    return item.text;
                  } else if (item.type === "file-data") {
                    return `[File: ${item.mediaType}]`;
                  } else if (item.type === "file-url") {
                    return `[File URL: ${item.url}]`;
                  } else {
                    return `[Unknown content type]`;
                  }
                })
                .join("\n");
            }

            result.push({
              role: "user",
              content: `Tool "${part.toolName}" (id: ${part.toolCallId}) returned:\n${resultText}`,
            });
          }
        }
        break;
    }
  }

  return result;
}

export class LlamaCppLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3" as const;
  readonly provider = "llama.cpp";
  readonly modelId: string;

  /**
   * Supported URL patterns - empty since we only support local files
   */
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private modelHandle: number | null = null;
  private readonly config: LlamaCppModelConfig;
  private initPromise: Promise<void> | null = null;

  constructor(config: LlamaCppModelConfig) {
    this.config = config;
    this.modelId = config.modelPath;
  }

  private async ensureModelLoaded(): Promise<number> {
    if (this.modelHandle !== null && isModelLoaded(this.modelHandle)) {
      return this.modelHandle;
    }

    if (this.initPromise) {
      await this.initPromise;
      if (this.modelHandle !== null) {
        return this.modelHandle;
      }
    }

    this.initPromise = (async () => {
      const options: LoadModelOptions = {
        modelPath: this.config.modelPath,
        contextSize: this.config.contextSize ?? 2048,
        gpuLayers: this.config.gpuLayers ?? 99,
        threads: this.config.threads ?? 4,
        debug: this.config.debug ?? false,
        chatTemplate: this.config.chatTemplate ?? "auto",
      };

      this.modelHandle = await loadModel(options);
    })();

    await this.initPromise;
    this.initPromise = null;

    if (this.modelHandle === null) {
      throw new Error("Failed to load model");
    }

    return this.modelHandle;
  }

  async dispose(): Promise<void> {
    if (this.modelHandle !== null) {
      unloadModel(this.modelHandle);
      this.modelHandle = null;
    }
  }

  async doGenerate(
    options: LanguageModelV3CallOptions
  ): Promise<LanguageModelV3GenerateResult> {
    const handle = await this.ensureModelLoaded();

    // Extract function tools from the tools array
    const functionTools =
      options.tools?.filter(
        (t): t is LanguageModelV3FunctionTool => t.type === "function"
      ) ?? [];

    const hasTools = functionTools.length > 0;

    const messages = convertMessages(
      options.prompt,
      hasTools ? functionTools : undefined
    );

    // Convert JSON schema to GBNF grammar if structured output is requested
    // or generate tool call grammar if tools are provided
    let grammar: string | undefined;
    if (
      options.responseFormat?.type === "json" &&
      options.responseFormat.schema
    ) {
      grammar = convertJsonSchemaToGrammar(
        options.responseFormat.schema as JSONSchema7
      );
    } else if (hasTools && options.toolChoice?.type !== "none") {
      // Generate grammar for tool calls
      grammar = generateToolCallGrammar(functionTools);
    }

    const generateOptions: GenerateOptions = {
      messages,
      maxTokens: options.maxOutputTokens ?? 2048,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      stopSequences: options.stopSequences,
      grammar,
    };

    const result = await generate(handle, generateOptions);

    const warnings: SharedV3Warning[] = [];
    const content: LanguageModelV3Content[] = [];
    let finishReason = convertFinishReason(result.finishReason);

    // Try to parse tool calls if tools were provided
    if (hasTools && options.toolChoice?.type !== "none") {
      const toolCalls = parseToolCalls(result.text);

      if (toolCalls && toolCalls.length > 0) {
        // Add tool calls to content
        for (const toolCall of toolCalls) {
          content.push({
            type: "tool-call",
            toolCallId: toolCall.id || generateToolCallId(),
            toolName: toolCall.name,
            input: JSON.stringify(toolCall.arguments),
          });
        }

        // Set finish reason to tool-calls
        finishReason = {
          unified: "tool-calls",
          raw: "tool-calls",
        };
      } else {
        // No valid tool calls found, return as text
        content.push({
          type: "text",
          text: result.text,
          providerMetadata: undefined,
        });
      }
    } else {
      // No tools, return text content
      content.push({
        type: "text",
        text: result.text,
        providerMetadata: undefined,
      });
    }

    return {
      content,
      finishReason,
      usage: convertUsage(result.promptTokens, result.completionTokens),
      warnings,
      request: {
        body: generateOptions,
      },
    };
  }

  async doStream(
    options: LanguageModelV3CallOptions
  ): Promise<LanguageModelV3StreamResult> {
    const handle = await this.ensureModelLoaded();

    // Extract function tools from the tools array
    const functionTools =
      options.tools?.filter(
        (t): t is LanguageModelV3FunctionTool => t.type === "function"
      ) ?? [];

    const hasTools = functionTools.length > 0;

    const messages = convertMessages(
      options.prompt,
      hasTools ? functionTools : undefined
    );

    // Convert JSON schema to GBNF grammar if structured output is requested
    // or generate tool call grammar if tools are provided
    let grammar: string | undefined;
    if (
      options.responseFormat?.type === "json" &&
      options.responseFormat.schema
    ) {
      grammar = convertJsonSchemaToGrammar(
        options.responseFormat.schema as JSONSchema7
      );
    } else if (hasTools && options.toolChoice?.type !== "none") {
      // Generate grammar for tool calls
      grammar = generateToolCallGrammar(functionTools);
    }

    const generateOptions: GenerateOptions = {
      messages,
      maxTokens: options.maxOutputTokens ?? 2048,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 40,
      stopSequences: options.stopSequences,
      grammar,
    };

    const textId = crypto.randomUUID();

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start: async (controller) => {
        try {
          // Emit stream start
          controller.enqueue({
            type: "stream-start",
            warnings: [],
          });

          // Collect the full text for tool call parsing
          let fullText = "";

          // Emit text start
          controller.enqueue({
            type: "text-start",
            id: textId,
          });

          const result = await generateStream(
            handle,
            generateOptions,
            (token) => {
              fullText += token;
              controller.enqueue({
                type: "text-delta",
                id: textId,
                delta: token,
              });
            }
          );

          // Emit text end
          controller.enqueue({
            type: "text-end",
            id: textId,
          });

          // Check for tool calls if tools were provided
          let finishReason = convertFinishReason(result.finishReason);

          if (hasTools && options.toolChoice?.type !== "none") {
            const toolCalls = parseToolCalls(fullText);

            if (toolCalls && toolCalls.length > 0) {
              // Emit tool call events
              for (const toolCall of toolCalls) {
                const toolCallId = toolCall.id || generateToolCallId();

                controller.enqueue({
                  type: "tool-call",
                  toolCallId,
                  toolName: toolCall.name,
                  input: JSON.stringify(toolCall.arguments),
                });
              }

              // Set finish reason to tool-calls
              finishReason = {
                unified: "tool-calls",
                raw: "tool-calls",
              };
            }
          }

          // Emit finish
          controller.enqueue({
            type: "finish",
            finishReason,
            usage: convertUsage(result.promptTokens, result.completionTokens),
          });

          controller.close();
        } catch (error) {
          controller.enqueue({
            type: "error",
            error,
          });
          controller.close();
        }
      },
    });

    return {
      stream,
      request: {
        body: generateOptions,
      },
    };
  }
}
