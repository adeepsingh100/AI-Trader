"use client";

import Link from "next/link";
import { useSearchParams, usePathname } from "next/navigation";
import type { Mode } from "@/lib/types";

export function useMode(): Mode {
  const params = useSearchParams();
  return params.get("mode") === "real" ? "real" : "paper";
}

export function ModeToggle() {
  const mode = useMode();
  const pathname = usePathname();

  return (
    <div
      className="inline-flex rounded-md overflow-hidden text-sm shadow-sm"
      style={{ border: "1px solid var(--border)" }}
    >
      {(["paper", "real"] as const).map((m) => (
        <Link
          key={m}
          href={`${pathname}?mode=${m}`}
          className="px-3 py-1 font-medium transition-colors"
          style={
            m === mode
              ? { background: "var(--text-primary)", color: "#fff" }
              : { background: "var(--surface-1)", color: "var(--text-secondary)" }
          }
        >
          {m === "paper" ? "Paper" : "Real"}
        </Link>
      ))}
    </div>
  );
}
