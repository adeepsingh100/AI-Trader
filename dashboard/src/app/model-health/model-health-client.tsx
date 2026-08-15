"use client";

import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { supabase } from "@/lib/supabase";
import { aggregateModelUsage } from "@/lib/modelUsageStats";
import { CHROME, SERIES } from "@/lib/palette";
import type { ModelUsage } from "@/lib/types";

export default function ModelHealthClient() {
  const [events, setEvents] = useState<ModelUsage[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    supabase
      .from("model_usage")
      .select("*")
      .order("timestamp", { ascending: false })
      .limit(500)
      .then(({ data, error }) => {
        if (cancelled) return;
        setError(error ? error.message : null);
        setEvents(error ? null : (data ?? []));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const stats = useMemo(() => aggregateModelUsage(events ?? []), [events]);
  const chartData = stats.map((s) => ({
    model: s.model,
    "success %": Math.round(s.successRate * 1000) / 10,
    "fallback %": Math.round(s.fallbackRate * 1000) / 10,
  }));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Model health</h1>

      {error && (
        <p className="text-sm rounded-lg p-3" style={{ background: "#fdecea", color: "var(--status-critical)" }}>
          Query failed: {error}
        </p>
      )}
      {!error && events === null && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
      {!error && events?.length === 0 && (
        <p style={{ color: "var(--text-muted)" }}>No model_usage recorded yet.</p>
      )}

      {stats.length > 0 && (
        <>
          <div
            className="rounded-xl p-4 shadow-sm"
            style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
          >
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHROME.gridline} vertical={false} />
                <XAxis
                  dataKey="model"
                  fontSize={12}
                  stroke={CHROME.axis}
                  tick={{ fill: CHROME.textMuted }}
                  tickLine={false}
                />
                <YAxis
                  fontSize={12}
                  unit="%"
                  stroke={CHROME.axis}
                  tick={{ fill: CHROME.textMuted }}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: CHROME.surface,
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 13,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: CHROME.textSecondary }} />
                <Bar dataKey="success %" fill={SERIES.blue} radius={[4, 4, 0, 0]} />
                <Bar dataKey="fallback %" fill={SERIES.red} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

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
                  {["Model", "Calls", "Success rate", "Fallback rate", "Avg latency"].map((h) => (
                    <th key={h} className="px-3 py-2.5 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stats.map((s) => (
                  <tr
                    key={s.model}
                    className="hover:bg-black/[0.02] transition-colors"
                    style={{ borderBottom: "1px solid var(--gridline)" }}
                  >
                    <td className="px-3 py-2.5 font-medium">{s.model}</td>
                    <td className="px-3 py-2.5 tabular-nums">{s.calls}</td>
                    <td className="px-3 py-2.5 tabular-nums">{(s.successRate * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2.5 tabular-nums">{(s.fallbackRate * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2.5 tabular-nums">{s.avgLatencyMs.toFixed(0)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
