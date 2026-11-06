import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Port 5173 and 8000 are used by another project on this machine.
export default defineConfig(({ command }) => ({
  plugins: [react()],
  // In production Django serves the built files, from /static/.
  base: command === 'build' ? '/static/' : '/',
  server: {
    port: 5174,
    host: '127.0.0.1',
    proxy: { '/api': 'http://127.0.0.1:8010' },
  },
}))
