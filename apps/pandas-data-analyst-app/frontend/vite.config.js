import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5175,
    proxy: {
      '/api': { target: 'http://localhost:8010', changeOrigin: true },
      '/health': { target: 'http://localhost:8010', changeOrigin: true },
      '/static': { target: 'http://localhost:8010', changeOrigin: true },
    }
  }
})


