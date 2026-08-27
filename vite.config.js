import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // The sandbox serves this through an e2b preview host; allow any Host
    // header so the proxied preview is not rejected.
    allowedHosts: true,
    hmr: { clientPort: 443, protocol: 'wss' },
  },
  worker: { format: 'es' },
  build: { target: 'es2022', sourcemap: true },
});
