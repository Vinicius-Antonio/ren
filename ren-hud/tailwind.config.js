/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        hermes: {
          cyan: "#00f0ff",
          blue: "#0066ff",
          green: "#00ff88",
        },
      },
    },
  },
  plugins: [],
};

