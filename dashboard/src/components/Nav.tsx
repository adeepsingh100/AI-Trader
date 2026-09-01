"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { signOut } from "firebase/auth";
import { auth } from "@/lib/firebase";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/trades", label: "Trades" },
  { href: "/evolution", label: "Evolution" },
  { href: "/learning", label: "Learning" },
  { href: "/model-health", label: "Model health" },
  { href: "/config", label: "Config" },
];

export function Nav() {
  const pathname = usePathname();
  const router = useRouter();

  // proxy.ts gates every route (including /sign-in redirect targets) on
  // the session cookie, not the client SDK's own auth state — so signing
  // out means clearing both: the client SDK session (so a stale client
  // doesn't silently re-mint a token) and the cookie via /api/session
  // (what proxy.ts actually checks).
  async function handleSignOut() {
    await signOut(auth);
    await fetch("/api/session", { method: "DELETE" });
    router.push("/sign-in");
    router.refresh();
  }

  if (pathname === "/sign-in") return null;

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
        <button
          onClick={handleSignOut}
          className="ml-auto text-sm hover:underline"
          style={{ color: "var(--text-muted)" }}
        >
          Sign out
        </button>
      </div>
    </header>
  );
}
