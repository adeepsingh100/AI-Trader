"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchJson } from "@/lib/api";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import { CHROME, SERIES } from "@/lib/palette";
import type { DailyPnl, StrategyVersion } from "@/lib/types";

interface EvolutionData {
  versions: StrategyVersion[];
  dailyPnl: DailyPnl[];
}

export default function EvolutionClient() {
  const mode = useMode();
  const [versions, setVersions] = useState<StrategyVersion[] | null>(null);
  const [dailyPnl, setDailyPnl] = useState<DailyPnl[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await fetchJson<EvolutionData>(`/api/evolution?mode=${mode}`);
        if (cancelled) return;
        setError(null);
        setVersions(data.versions);
        setDailyPnl(data.dailyPnl);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setVersions(null);
        setDailyPnl(null);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [mode]);

  const chartData = useMemo(() => {
    let cumulative = 0;
    return (dailyPnl ?? []).map((row) => {
      cumulative += row.realized_pnl;
      return { date: row.date, cumulative_pnl: cumulative };
    });
  }, [dailyPnl]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Evolution</h1>
        <ModeToggle />
      </div>

      {error && (
        <p className="text-sm rounded-lg p-3" style={{ background: "#fdecea", color: "var(--status-critical)" }}>
          Query failed: {error}
        </p>
      )}

      <div
        className="rounded-xl p-4 shadow-sm"
        style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
      >
        <h2 className="text-xs uppercase tracking-wide mb-3" style={{ color: "var(--text-muted)" }}>
          Cumulative realized PnL ({mode})
        </h2>
        {chartData.length === 0 ? (
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            No daily_pnl history yet.
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHROME.gridline} vertical={false} />
              <XAxis
                dataKey="date"
                fontSize={12}
                stroke={CHROME.axis}
                tick={{ fill: CHROME.textMuted }}
                tickLine={false}
              />
              <YAxis fontSize={12} stroke={CHROME.axis} tick={{ fill: CHROME.textMuted }} tickLine={false} />
              <Tooltip
                contentStyle={{
                  background: CHROME.surface,
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  fontSize: 13,
                }}
              />
              <Line
                type="monotone"
                dataKey="cumulative_pnl"
                stroke={SERIES.blue}
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div>
        <h2 className="text-xs uppercase tracking-wide mb-2" style={{ color: "var(--text-muted)" }}>
          Version history
        </h2>
        {versions === null && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
        {versions && (
          <div
            className="overflow-x-auto rounded-xl shadow-sm"
            style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
          >
            <table className="w-full text-sm">
              <thead>
                <tr
                  className="text-left text-xs uppercase tracking-wide"
                  style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--gridline)" }}
                >
                  {["Version", "Status", "Notes", "Created"].map((h) => (
                    <th key={h} className="px-3 py-2.5 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {versions.map((v) => (
                  <tr
                    key={v.id}
                    className="hover:bg-black/[0.02] transition-colors"
                    style={{ borderBottom: "1px solid var(--gridline)" }}
                  >
                    <td className="px-3 py-2.5 font-medium tabular-nums">v{v.version_number}</td>
                    <td className="px-3 py-2.5">
                      {v.promoted_to_real ? (
                        <span
                          className="text-xs font-medium px-1.5 py-0.5 rounded"
                          style={{ color: "var(--status-good)", background: "#eafaea" }}
                        >
                          real
                        </span>
                      ) : (
                        <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                          paper-only
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2.5" style={{ color: "var(--text-secondary)" }}>
                      {v.notes}
                    </td>
                    <td className="px-3 py-2.5 whitespace-nowrap" style={{ color: "var(--text-secondary)" }}>
                      {new Date(v.created_at).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
