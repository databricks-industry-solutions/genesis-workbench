import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

const BACKEND = process.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // Plotly is ~3.5 MB and dominates the main chunk; split it (and a few
    // other heavy vendor libs) into their own chunks so the initial page
    // load only pulls what's needed for the current route.
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('plotly.js')) return 'plotly'
            if (id.includes('molstar')) return 'molstar'
            if (id.includes('@tanstack')) return 'tanstack'
            if (id.includes('react-markdown') || id.includes('remark')) return 'markdown'
          }
        },
      },
    },
    // Bump the warning threshold so the plotly chunk (a single intentional
    // ~1 MB chunk) doesn't fire on every build.
    chunkSizeWarningLimit: 1500,
  },
})
