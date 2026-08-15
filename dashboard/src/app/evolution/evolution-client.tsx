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
import { supabase } from "@/lib/supabase";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import type { DailyPnl, StrategyVersion } from "@/lib/types";

export default function EvolutionClient() {
  const mode = useMode();
  const [versions, setVersions] = useState<StrategyVersion[] | null>(null);
  const [dailyPnl, setDailyPnl] = useState<DailyPnl[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const [
        { data: versionRows, error: versionsError },
        { data: pnlRows, error: pnlError },
      ] = await Promise.all([
        supabase.from("strategy_versions").select("*").order("version_number", { ascending: false }),
        supabase.from("daily_pnl").select("*").eq("mode", mode).order("date", { ascending: true }),
      ]);
      if (cancelled) return;
      const err = versionsError ?? pnlError;
      setError(err ? err.message : null);
      setVersions(err ? null : (versionRows ?? []));
      setDailyPnl(err ? null : (pnlRows ?? []));
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
        <h1 className="text-xl font-semibold">Evolution</h1>
        <ModeToggle />
      </div>

      {error && (
        <p className="text-red-600 text-sm rounded border border-red-200 bg-red-50 p-3">
          Query failed: {error}
        </p>
      )}

      <div className="rounded-lg border border-neutral-200 bg-white p-4">
        <h2 className="text-sm font-medium text-neutral-500 mb-2">
          Cumulative realized PnL ({mode})
        </h2>
        {chartData.length === 0 ? (
          <p className="text-neutral-500 text-sm">No daily_pnl history yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Line type="monotone" dataKey="cumulative_pnl" stroke="#0f766e" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div>
        <h2 className="text-sm font-medium text-neutral-500 mb-2">Version history</h2>
        {versions === null && <p className="text-neutral-500">Loading…</p>}
        {versions && (
          <div className="overflow-x-auto rounded-lg border border-neutral-200 bg-white">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-neutral-500 border-b border-neutral-200">
                  {["Version", "Status", "Notes", "Created"].map((h) => (
                    <th key={h} className="px-3 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {versions.map((v) => (
                  <tr key={v.id} className="border-b border-neutral-100 last:border-0">
                    <td className="px-3 py-2 font-medium">v{v.version_number}</td>
                    <td className="px-3 py-2">
                      {v.promoted_to_real ? (
                        <span className="text-emerald-600">real</span>
                      ) : (
                        "paper-only"
                      )}
                    </td>
                    <td className="px-3 py-2 text-neutral-600">{v.notes}</td>
                    <td className="px-3 py-2 whitespace-nowrap">
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
