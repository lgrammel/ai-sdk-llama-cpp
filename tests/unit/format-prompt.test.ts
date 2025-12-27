import { describe, it, expect } from "vitest";
import { formatPrompt } from "../../src/llama-cpp-language-model.js";
import type { LanguageModelV3Message } from "@ai-sdk/provider";

describe("formatPrompt", () => {
  describe("system messages", () => {
    it("formats system message as user turn with Gemma format", () => {
      const messages: LanguageModelV3Message[] = [
        { role: "system", content: "You are a helpful assistant." },
      ];

      const result = formatPrompt(messages);

      expect(result).toContain("<start_of_turn>user");
      expect(result).toContain("You are a helpful assistant.");
      expect(result).toContain("<end_of_turn>");
      expect(result.endsWith("<start_of_turn>model\n")).toBe(true);
    });
  });

  describe("user messages", () => {
    it("formats user message with text content", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "Hello, how are you?" }],
        },
      ];

      const result = formatPrompt(messages);

      expect(result).toBe(
        "<start_of_turn>user\nHello, how are you?<end_of_turn>\n<start_of_turn>model\n"
      );
    });

    it("concatenates multiple text parts in user message", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "user",
          content: [
            { type: "text", text: "First part. " },
            { type: "text", text: "Second part." },
          ],
        },
      ];

      const result = formatPrompt(messages);

      expect(result).toContain("First part. Second part.");
    });

    it("ignores non-text parts in user message", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "user",
          content: [
            { type: "text", text: "Hello" },
            // File parts are not supported, should be ignored
            {
              type: "file",
              url: "data:image/png;base64,abc",
              mediaType: "image/png",
            } as any,
          ],
        },
      ];

      const result = formatPrompt(messages);

      expect(result).toContain("Hello");
      expect(result).not.toContain("data:image");
    });
  });

  describe("assistant messages", () => {
    it("formats assistant message with model turn tag", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "assistant",
          content: [{ type: "text", text: "I am doing well, thank you!" }],
        },
      ];

      const result = formatPrompt(messages);

      expect(result).toContain("<start_of_turn>model");
      expect(result).toContain("I am doing well, thank you!");
      expect(result).toContain("<end_of_turn>");
    });

    it("concatenates multiple text parts in assistant message", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "assistant",
          content: [
            { type: "text", text: "Part one. " },
            { type: "text", text: "Part two." },
          ],
        },
      ];

      const result = formatPrompt(messages);

      expect(result).toContain("Part one. Part two.");
    });
  });

  describe("tool messages", () => {
    it("ignores tool messages (not supported)", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_123",
              toolName: "get_weather",
              result: { temperature: 72 },
            },
          ],
        },
      ];

      const result = formatPrompt(messages);

      // Should only have the model prefix since tool messages are ignored
      expect(result).toBe("<start_of_turn>model\n");
    });
  });

  describe("multi-turn conversations", () => {
    it("formats a complete conversation with system, user, and assistant", () => {
      const messages: LanguageModelV3Message[] = [
        { role: "system", content: "You are helpful." },
        { role: "user", content: [{ type: "text", text: "Hi!" }] },
        {
          role: "assistant",
          content: [{ type: "text", text: "Hello there!" }],
        },
        {
          role: "user",
          content: [{ type: "text", text: "How are you?" }],
        },
      ];

      const result = formatPrompt(messages);

      // Check order and structure
      const systemIdx = result.indexOf("You are helpful.");
      const userHiIdx = result.indexOf("Hi!");
      const assistantIdx = result.indexOf("Hello there!");
      const userHowIdx = result.indexOf("How are you?");

      expect(systemIdx).toBeLessThan(userHiIdx);
      expect(userHiIdx).toBeLessThan(assistantIdx);
      expect(assistantIdx).toBeLessThan(userHowIdx);

      // Should end with model turn for generation
      expect(result.endsWith("<start_of_turn>model\n")).toBe(true);
    });

    it("handles multiple user-assistant exchanges", () => {
      const messages: LanguageModelV3Message[] = [
        { role: "user", content: [{ type: "text", text: "Question 1" }] },
        { role: "assistant", content: [{ type: "text", text: "Answer 1" }] },
        { role: "user", content: [{ type: "text", text: "Question 2" }] },
        { role: "assistant", content: [{ type: "text", text: "Answer 2" }] },
        { role: "user", content: [{ type: "text", text: "Question 3" }] },
      ];

      const result = formatPrompt(messages);

      // Count occurrences of turn markers
      const userTurns = (result.match(/<start_of_turn>user/g) || []).length;
      const modelTurns = (result.match(/<start_of_turn>model/g) || []).length;

      expect(userTurns).toBe(3);
      // 2 assistant responses + 1 final model turn for generation
      expect(modelTurns).toBe(3);
    });
  });

  describe("edge cases", () => {
    it("handles empty messages array", () => {
      const messages: LanguageModelV3Message[] = [];

      const result = formatPrompt(messages);

      expect(result).toBe("<start_of_turn>model\n");
    });

    it("handles empty text content", () => {
      const messages: LanguageModelV3Message[] = [
        { role: "user", content: [{ type: "text", text: "" }] },
      ];

      const result = formatPrompt(messages);

      expect(result).toContain("<start_of_turn>user\n<end_of_turn>");
    });

    it("preserves newlines in content", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "Line 1\nLine 2\nLine 3" }],
        },
      ];

      const result = formatPrompt(messages);

      expect(result).toContain("Line 1\nLine 2\nLine 3");
    });

    it("preserves special characters in content", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "user",
          content: [
            { type: "text", text: "Special chars: <>&\"'`${}[]" },
          ],
        },
      ];

      const result = formatPrompt(messages);

      expect(result).toContain("Special chars: <>&\"'`${}[]");
    });
  });
});

