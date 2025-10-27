import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: env.VITE_API_URL
        ? undefined
        : {
            "/api": "http://localhost:8000",
            "/ws": {
              target: "ws://localhost:8000",
              ws: true
            }
          }
    },
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "src")
      }
    }
  };
});
