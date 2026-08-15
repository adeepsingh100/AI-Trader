// Validated palette (dataviz skill) — light-only, hex here since Recharts
// props need real values, not CSS vars. Keep in sync with globals.css.

export const CHROME = {
  surface: "#fcfcfb",
  gridline: "#e1e0d9",
  axis: "#c3c2b7",
  textSecondary: "#52514e",
  textMuted: "#898781",
} as const;

// Categorical slots 1 (blue) and 8 (red) — the only pair of the 8-slot
// order that clears CVD separation cleanly with no warnings; status
// good/critical failed that check for this specific pairing.
export const SERIES = {
  blue: "#2a78d6",
  red: "#e34948",
} as const;

// Fixed, never themed — reserved for state, always paired with a text label.
export const STATUS = {
  good: "#0ca30c",
  warning: "#fab219",
  critical: "#d03b3b",
} as const;
