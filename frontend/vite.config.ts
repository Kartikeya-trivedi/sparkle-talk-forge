import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    hmr: {
      overlay: false,
    },
    proxy: {
      "/api": {
        // KT GPT v2 — Production RAG Server (Gemma 4 26B / Llama 3.1 8B)
        target: "https://mindrix--ktgpt-rag-server-ragserver-serve.modal.run",
        // KT GPT v1 — Original KTGPT custom model (legacy, kept for reference)
        // target: "https://kartikeyatrivedi4oct2004--ktgpt-server-ktgptserver-serve-dev.modal.run",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    dedupe: ["react", "react-dom", "react/jsx-runtime", "react/jsx-dev-runtime", "@tanstack/react-query", "@tanstack/query-core"],
  },
}));
