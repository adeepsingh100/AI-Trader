"use client";

import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { supabase } from "@/lib/supabase";
import { aggregateModelUsage } from "@/lib/modelUsageStats";
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
      <h1 className="text-xl font-semibold">Model health</h1>

      {error && (
        <p className="text-red-600 text-sm rounded border border-red-200 bg-red-50 p-3">
          Query failed: {error}
        </p>
      )}
      {!error && events === null && <p className="text-neutral-500">Loading…</p>}
      {!error && events?.length === 0 && <p className="text-neutral-500">No model_usage recorded yet.</p>}

      {stats.length > 0 && (
        <>
          <div className="rounded-lg border border-neutral-200 bg-white p-4">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" fontSize={12} />
                <YAxis fontSize={12} unit="%" />
                <Tooltip />
                <Legend />
                <Bar dataKey="success %" fill="#0f766e" />
                <Bar dataKey="fallback %" fill="#b45309" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="overflow-x-auto rounded-lg border border-neutral-200 bg-white">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-neutral-500 border-b border-neutral-200">
                  {["Model", "Calls", "Success rate", "Fallback rate", "Avg latency"].map((h) => (
                    <th key={h} className="px-3 py-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stats.map((s) => (
                  <tr key={s.model} className="border-b border-neutral-100 last:border-0">
                    <td className="px-3 py-2 font-medium">{s.model}</td>
                    <td className="px-3 py-2">{s.calls}</td>
                    <td className="px-3 py-2">{(s.successRate * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2">{(s.fallbackRate * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2">{s.avgLatencyMs.toFixed(0)} ms</td>
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
