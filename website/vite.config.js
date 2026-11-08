import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import compactScale from './scripts/compact-scale.mjs';

// base './' keeps every emitted URL relative to index.html, so the dist
// folder can be dropped into any static host (sub-path, OSS, GitHub Pages)
// without rewriting asset paths.
export default defineConfig({
  plugins: [vue()],
  base: './',
  css: { postcss: { plugins: [compactScale()] } },
  server: {
    port: 5173,
    proxy: {
      // When the backend aggregation service is available, the app picks up
      // live /api/bundle automatically (src/services/api.js). This proxy
      // wires the dev server to the default backend port.
      '/api': { target: 'http://127.0.0.1:8787', changeOrigin: true }
    }
  }
});
