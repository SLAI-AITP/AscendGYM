// Minimal stand-in for the backend aggregation service (README §6).
//
// Serves `GET /api/bundle` from `src/data/operator_bundle.json` — the snapshot
// `npm run bundle:operator` writes from `src/data/eval_datasets_summary/` — on
// 127.0.0.1:8787, so the Vite dev proxy (vite.config.js) has a live target and
// the frontend runs in `source === 'live'` mode.
//
// The browser does NOT need this: it reads the dataset folder directly (see
// src/services/datasets.js). This mock exists only so the `live` code path stays
// exercisable during frontend development, and so the dev console stays free of
// ECONNREFUSED proxy errors.
//
//   npm run dev:api     # listens on 127.0.0.1:8787 (override with PORT env)
//
// The bundle file is re-read on every request, so re-running
// `npm run bundle:operator` is picked up without restarting this server.

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BUNDLE_PATH = path.resolve(__dirname, '../src/data/operator_bundle.json');
const HOST = '127.0.0.1';
const PORT = Number(process.env.PORT || 8787);

function sendJson(res, status, payload) {
  // Buffer payloads (raw file bytes) pass through untouched; objects are
  // stringified once. Never JSON.stringify a Buffer — that yields a
  // {"type":"Buffer","data":[...]} envelope.
  const body = Buffer.isBuffer(payload) ? payload : Buffer.from(JSON.stringify(payload), 'utf8');
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': body.length,
    'Cache-Control': 'no-store'
  });
  res.end(body);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', `http://${HOST}:${PORT}`);

  if (req.method === 'GET' && url.pathname === '/api/bundle') {
    try {
      const body = await readFile(BUNDLE_PATH);
      sendJson(res, 200, body);
    } catch (err) {
      sendJson(res, 500, { error: `failed to read bundle: ${err.message}` });
    }
    return;
  }

  if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/healthz')) {
    sendJson(res, 200, { ok: true, service: 'ascendgym-ux-v2-mock-api' });
    return;
  }

  sendJson(res, 404, { error: `no such endpoint: ${req.method} ${url.pathname}` });
});

server.listen(PORT, HOST, () => {
  console.log(`[mock-api] GET /api/bundle -> ${path.relative(process.cwd(), BUNDLE_PATH)}`);
  console.log(`[mock-api] listening on http://${HOST}:${PORT}`);
});

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => {
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 1000).unref();
  });
}
