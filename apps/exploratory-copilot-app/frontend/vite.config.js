import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8008',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8008',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://localhost:8008',
        changeOrigin: true,
      }
    }
  }
})

