import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          900: "#0f172a",
          800: "#1e293b",
          700: "#334155"
        },
        accent: {
          500: "#38bdf8",
          400: "#60a5fa"
        }
      }
    }
  },
  plugins: []
};

export default config;
