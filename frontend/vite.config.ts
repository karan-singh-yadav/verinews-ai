import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In Docker the dev server must reach the backend by service name: VITE_PROXY_TARGET=http://backend:8000
const apiTarget = process.env.VITE_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { "/api": apiTarget } },
  test: { environment: "jsdom", globals: true, setupFiles: "./src/test/setup.ts", css: false },
});
