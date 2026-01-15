import { describe, it, expect } from "vitest";
import { convertMessages } from "../../src/llama-cpp-language-model.js";
import type { LanguageModelV3Message } from "@ai-sdk/provider";

describe("convertMessages", () => {
  describe("system messages", () => {
    it("converts system message to role/content format", () => {
      const messages: LanguageModelV3Message[] = [
        { role: "system", content: "You are a helpful assistant." },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "system", content: "You are a helpful assistant." },
      ]);
    });
  });

  describe("user messages", () => {
    it("converts user message with text content", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "Hello, how are you?" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "user", content: "Hello, how are you?" },
      ]);
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

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "user", content: "First part. Second part." },
      ]);
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

      const result = convertMessages(messages);

      expect(result).toEqual([{ role: "user", content: "Hello" }]);
    });
  });

  describe("assistant messages", () => {
    it("converts assistant message to role/content format", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "assistant",
          content: [{ type: "text", text: "I am doing well, thank you!" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "assistant", content: "I am doing well, thank you!" },
      ]);
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

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "assistant", content: "Part one. Part two." },
      ]);
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

      const result = convertMessages(messages);

      // Should return empty array since tool messages are ignored
      expect(result).toEqual([]);
    });
  });

  describe("multi-turn conversations", () => {
    it("converts a complete conversation with system, user, and assistant", () => {
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

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hi!" },
        { role: "assistant", content: "Hello there!" },
        { role: "user", content: "How are you?" },
      ]);
    });

    it("handles multiple user-assistant exchanges", () => {
      const messages: LanguageModelV3Message[] = [
        { role: "user", content: [{ type: "text", text: "Question 1" }] },
        { role: "assistant", content: [{ type: "text", text: "Answer 1" }] },
        { role: "user", content: [{ type: "text", text: "Question 2" }] },
        { role: "assistant", content: [{ type: "text", text: "Answer 2" }] },
        { role: "user", content: [{ type: "text", text: "Question 3" }] },
      ];

      const result = convertMessages(messages);

      expect(result).toHaveLength(5);
      expect(result[0]).toEqual({ role: "user", content: "Question 1" });
      expect(result[1]).toEqual({ role: "assistant", content: "Answer 1" });
      expect(result[2]).toEqual({ role: "user", content: "Question 2" });
      expect(result[3]).toEqual({ role: "assistant", content: "Answer 2" });
      expect(result[4]).toEqual({ role: "user", content: "Question 3" });
    });
  });

  describe("edge cases", () => {
    it("handles empty messages array", () => {
      const messages: LanguageModelV3Message[] = [];

      const result = convertMessages(messages);

      expect(result).toEqual([]);
    });

    it("handles empty text content", () => {
      const messages: LanguageModelV3Message[] = [
        { role: "user", content: [{ type: "text", text: "" }] },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([{ role: "user", content: "" }]);
    });

    it("preserves newlines in content", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "Line 1\nLine 2\nLine 3" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "user", content: "Line 1\nLine 2\nLine 3" },
      ]);
    });

    it("preserves special characters in content", () => {
      const messages: LanguageModelV3Message[] = [
        {
          role: "user",
          content: [{ type: "text", text: "Special chars: <>&\"'`${}[]" }],
        },
      ];

      const result = convertMessages(messages);

      expect(result).toEqual([
        { role: "user", content: "Special chars: <>&\"'`${}[]" },
      ]);
    });
  });
});
