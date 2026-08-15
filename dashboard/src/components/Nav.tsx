import Link from "next/link";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/trades", label: "Trades" },
  { href: "/evolution", label: "Evolution" },
  { href: "/model-health", label: "Model health" },
  { href: "/config", label: "Config" },
];

export function Nav() {
  return (
    <header className="border-b border-neutral-200">
      <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-6">
        <span className="font-semibold">AI-Trader</span>
        <nav className="flex gap-4 text-sm">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href} className="text-neutral-600 hover:text-neutral-900">
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
