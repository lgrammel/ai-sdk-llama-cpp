const { execSync } = require('child_process');

if (process.platform !== 'darwin') {
  console.error('\n===========================================');
  console.error('ERROR: ai-sdk-llama-cpp only supports macOS');
  console.error('===========================================\n');
  console.error(`Detected platform: ${process.platform}`);
  console.error('This package requires macOS for native compilation.\n');
  process.exit(1);
}

console.log('Building native llama.cpp bindings for macOS...');
execSync('npx cmake-js compile', { stdio: 'inherit' });

