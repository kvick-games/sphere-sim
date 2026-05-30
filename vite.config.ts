import { defineConfig } from 'vite';

export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/sphere-sim/' : '/',
  server: {
    host: '127.0.0.1',
    port: 5180,
    allowedHosts: ['dreamatron.tail98fefd.ts.net'],
  },
}));
