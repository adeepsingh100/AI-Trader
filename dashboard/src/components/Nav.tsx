"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/trades", label: "Trades" },
  { href: "/evolution", label: "Evolution" },
  { href: "/model-health", label: "Model health" },
  { href: "/config", label: "Config" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <header className="border-b" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
      <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-6">
        <span className="font-semibold tracking-tight">AI-Trader</span>
        <nav className="flex gap-1 text-sm">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={
                  "px-3 py-1.5 rounded-md transition-colors " +
                  (active
                    ? "bg-neutral-900 text-white font-medium"
                    : "text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100")
                }
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
