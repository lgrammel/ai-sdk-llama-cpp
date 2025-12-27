import { createRequire } from 'node:module';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Load the native binding
const binding = require(join(__dirname, '..', 'build', 'Release', 'llama_binding.node')) as NativeBinding;

export interface LoadModelOptions {
  modelPath: string;
  gpuLayers?: number;
  contextSize?: number;
  threads?: number;
}

export interface GenerateOptions {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
}

export interface GenerateResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
  finishReason: 'stop' | 'length' | 'error';
}

interface NativeBinding {
  loadModel(options: LoadModelOptions, callback: (error: string | null, handle: number | null) => void): void;
  unloadModel(handle: number): boolean;
  generate(handle: number, options: GenerateOptions, callback: (error: string | null, result: GenerateResult | null) => void): void;
  generateStream(
    handle: number,
    options: GenerateOptions,
    tokenCallback: (token: string) => void,
    doneCallback: (error: string | null, result: GenerateResult | null) => void
  ): void;
  isModelLoaded(handle: number): boolean;
}

export function loadModel(options: LoadModelOptions): Promise<number> {
  return new Promise((resolve, reject) => {
    binding.loadModel(options, (error, handle) => {
      if (error) {
        reject(new Error(error));
      } else if (handle !== null) {
        resolve(handle);
      } else {
        reject(new Error('Failed to load model: unknown error'));
      }
    });
  });
}

export function unloadModel(handle: number): boolean {
  return binding.unloadModel(handle);
}

export function generate(handle: number, options: GenerateOptions): Promise<GenerateResult> {
  return new Promise((resolve, reject) => {
    binding.generate(handle, options, (error, result) => {
      if (error) {
        reject(new Error(error));
      } else if (result) {
        resolve(result);
      } else {
        reject(new Error('Failed to generate: unknown error'));
      }
    });
  });
}

export function generateStream(
  handle: number,
  options: GenerateOptions,
  onToken: (token: string) => void
): Promise<GenerateResult> {
  return new Promise((resolve, reject) => {
    binding.generateStream(
      handle,
      options,
      onToken,
      (error, result) => {
        if (error) {
          reject(new Error(error));
        } else if (result) {
          resolve(result);
        } else {
          reject(new Error('Failed to generate stream: unknown error'));
        }
      }
    );
  });
}

export function isModelLoaded(handle: number): boolean {
  return binding.isModelLoaded(handle);
}

