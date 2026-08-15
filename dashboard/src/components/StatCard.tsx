import { STATUS } from "@/lib/palette";

type Tone = "good" | "critical" | "neutral";

const TONE_COLOR: Record<Tone, string> = {
  good: STATUS.good,
  critical: STATUS.critical,
  neutral: "var(--text-secondary)",
};

export function StatCard({
  label,
  value,
  sub,
  tone = "neutral",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: Tone;
}) {
  return (
    <div
      className="rounded-xl p-4 shadow-sm"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
    >
      <div className="text-xs" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="flex items-center gap-1.5 mt-1">
        {tone !== "neutral" && (
          <span
            className="inline-block w-2 h-2 rounded-full shrink-0"
            style={{ background: TONE_COLOR[tone] }}
            aria-hidden
          />
        )}
        <span className="text-lg font-semibold tabular-nums" style={{ color: "var(--text-primary)" }}>
          {value}
        </span>
      </div>
      {sub && (
        <div className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
          {sub}
        </div>
      )}
    </div>
  );
}
