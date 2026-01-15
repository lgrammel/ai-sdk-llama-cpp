// Global test setup for ai-sdk-llama-cpp tests

// Environment variable for E2E tests
export const TEST_MODEL_PATH = process.env.TEST_MODEL_PATH;
export const TEST_EMBEDDING_PATH = process.env.TEST_EMBEDDING_PATH;

// Helper to check if E2E tests should run
export function shouldRunE2ETests(): boolean {
  return !!TEST_MODEL_PATH || !!TEST_EMBEDDING_PATH;
}

// Skip helper for conditional E2E tests
export function describeE2E(name: string, fn: () => void) {
  if (shouldRunE2ETests()) {
    describe(name, fn);
  } else {
    describe.skip(`${name} (skipped: TEST_MODEL_PATH not set)`, fn);
  }
}
