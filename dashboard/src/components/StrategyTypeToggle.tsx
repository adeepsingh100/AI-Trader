"use client";

import Link from "next/link";
import { useSearchParams, usePathname } from "next/navigation";

// Mirrors ModeToggle.tsx exactly (URL-derived, no context) — options are a
// small static array here rather than fetched from src/config.py's
// STRATEGY_PROFILES (the dashboard has no live view into Python config),
// same hardcode ModeToggle itself uses for ["paper","real"]. Revisit only
// once a 3rd strategy type actually ships.
const STRATEGY_TYPES = ["default", "swing"] as const;

export function useStrategyType(): string {
  const params = useSearchParams();
  return params.get("strategy_type") ?? "default";
}

export function StrategyTypeToggle() {
  const strategyType = useStrategyType();
  const pathname = usePathname();
  const params = useSearchParams();

  return (
    <div
      className="inline-flex rounded-md overflow-hidden text-sm shadow-sm"
      style={{ border: "1px solid var(--border)" }}
    >
      {STRATEGY_TYPES.map((st) => {
        const next = new URLSearchParams(params.toString());
        next.set("strategy_type", st);
        return (
          <Link
            key={st}
            href={`${pathname}?${next.toString()}`}
            className="px-3 py-1 font-medium capitalize transition-colors"
            style={
              st === strategyType
                ? { background: "var(--text-primary)", color: "#fff" }
                : { background: "var(--surface-1)", color: "var(--text-secondary)" }
            }
          >
            {st}
          </Link>
        );
      })}
    </div>
  );
}
