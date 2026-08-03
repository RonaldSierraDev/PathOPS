/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    // Tokens from docs/frontend-design-system.md -- the Foundry-style spec.
    // Monochrome by default; accent/status colors only at small scale.
    colors: {
      transparent: "transparent",
      current: "currentColor",
      ink: "#1C2127",
      "ink-soft": "#404854",
      "ink-mute": "#ABB3BF",
      paper: "#FFFFFF",
      "paper-tint": "#F6F7F9",
      hairline: "#DCE0E5",
      rail: "#111418",
      "rail-ink": "#8F99A8",
      accent: "#2D72D2",
      ok: "#238551",
      warn: "#C87619",
      danger: "#CD4246",
    },
    fontFamily: {
      sans: ["Inter", "system-ui", "sans-serif"],
      mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
    },
    extend: {
      borderRadius: {
        DEFAULT: "2px",
        card: "4px",
      },
      letterSpacing: {
        eyebrow: "0.06em",
      },
    },
  },
  plugins: [],
};
