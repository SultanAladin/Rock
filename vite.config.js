import { defineConfig } from 'vite';

export default defineConfig({
  define: { __BUILD__: JSON.stringify(new Date().toISOString().slice(5,16).replace('T',' ')) },
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // The sandbox serves this through an e2b preview host; allow any Host
    // header so the proxied preview is not rejected.
    allowedHosts: true,
    hmr: { clientPort: 443, protocol: 'wss' },
    // The preview is served through a proxy that will happily hand back a stale
    // bundle; force revalidation so a reload always picks up current code.
    headers: { 'Cache-Control': 'no-store, must-revalidate' },
  },
  worker: { format: 'es' },
  build: { target: 'es2022', sourcemap: true },
});
