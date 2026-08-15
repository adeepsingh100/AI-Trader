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
    <div className="inline-flex rounded-md border border-neutral-300 overflow-hidden text-sm">
      {(["paper", "real"] as const).map((m) => (
        <Link
          key={m}
          href={`${pathname}?mode=${m}`}
          className={
            "px-3 py-1 " +
            (m === mode ? "bg-neutral-900 text-white" : "bg-white text-neutral-700")
          }
        >
          {m === "paper" ? "Paper" : "Real"}
        </Link>
      ))}
    </div>
  );
}
