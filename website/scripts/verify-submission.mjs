import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { validateSubmitter } from '../src/lib/submission.js';

for (const input of [{}, { organization: 'Lab', email: 'a@b.org' }, { organization: ' ', email: 'a@b.org', name: 'Name' }, { organization: 'Lab', email: 'bad-email', name: 'Name' }, { organization: 'Lab', email: 'a @b.org', name: 'Name' }]) {
  assert.equal(validateSubmitter(input).ok, false);
}
const details = { organization: ' Lab ', name: ' Tester ', email: ' tester@example.org ' };
const normalized = { organization: 'Lab', name: 'Tester', email: 'tester@example.org' };
assert.deepEqual(validateSubmitter(details).value, normalized);

// Exercise the real upload store against isolated in-memory storage.
const memory = new Map();
const previousWindow = globalThis.window;
globalThis.window = { localStorage: {
  getItem: key => memory.get(key) ?? null,
  setItem: (key, value) => memory.set(key, value),
  removeItem: key => memory.delete(key)
} };
const tempRoot = fileURLToPath(new URL('../tmp/', import.meta.url));
fs.mkdirSync(tempRoot, { recursive: true });
const tempDir = fs.mkdtempSync(path.join(tempRoot, 'submission-check-'));
const output = path.join(tempDir, 'uploads.mjs');
try {
  await build({ entryPoints: [fileURLToPath(new URL('../src/store/uploads.js', import.meta.url))], outfile: output, bundle: true, platform: 'node', format: 'esm', external: ['vue'] });
  const store = await import(pathToFileURL(output).href);
  const raw = JSON.parse(fs.readFileSync(new URL('../src/data/eval_datasets_summary/claude-code-claude-opus-5.json', import.meta.url)));
  const original = JSON.stringify(raw);
  assert.equal(store.addUpload('test.json', raw, { ...details, email: 'bad-email' }).ok, false);
  assert.equal(store.uploads.files.length, 0);
  assert.equal(store.addUpload('test.json', raw, details).ok, true);
  const saved = JSON.parse(memory.get('ascendgym.uploads.v1')).files[0];
  assert.deepEqual(saved.submitter, normalized);
  assert.equal(JSON.stringify(saved.raw), original);
  assert.equal(JSON.stringify(raw), original);
  store.removeUpload('test.json');
  assert.equal(JSON.parse(memory.get('ascendgym.uploads.v1')).files.length, 0);
} finally {
  fs.rmSync(tempDir, { recursive: true, force: true });
  if (previousWindow === undefined) delete globalThis.window;
  else globalThis.window = previousWindow;
}
console.log('PASS: required submitter fields, email validation, normalized metadata storage, original JSON preservation and removal.');
