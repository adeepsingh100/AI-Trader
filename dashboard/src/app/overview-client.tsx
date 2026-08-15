"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { todayIst } from "@/lib/date";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import type { CapitalConfig, DailyPnl } from "@/lib/types";

export default function OverviewClient() {
  const mode = useMode();
  const [config, setConfig] = useState<CapitalConfig | null | undefined>(undefined);
  const [dailyPnl, setDailyPnl] = useState<DailyPnl | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const [
        { data: configRow, error: configError },
        { data: pnlRow, error: pnlError },
      ] = await Promise.all([
        supabase.from("capital_config").select("*").eq("mode", mode).maybeSingle(),
        supabase
          .from("daily_pnl")
          .select("*")
          .eq("mode", mode)
          .eq("date", todayIst())
          .maybeSingle(),
      ]);
      if (cancelled) return;
      const err = configError ?? pnlError;
      setError(err ? err.message : null);
      setConfig(err ? undefined : (configRow ?? null));
      setDailyPnl(pnlRow ?? null);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [mode]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Overview</h1>
        <ModeToggle />
      </div>

      {error && (
        <p className="text-red-600 text-sm rounded border border-red-200 bg-red-50 p-3">
          Query failed: {error}
        </p>
      )}
      {!error && config === undefined && <p className="text-neutral-500">Loading…</p>}
      {!error && config === null && (
        <p className="text-neutral-500">
          No capital_config row for {mode} yet — seed it with{" "}
          <code>python3 -m src.seed_config</code>.
        </p>
      )}

      {config && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            label="Trading status"
            value={config.paused ? "Stopped" : "Running"}
            sub={config.paused ? "no model calls, no new orders" : "cycles run on schedule"}
            tone={config.paused ? "negative" : "positive"}
          />
          <StatCard
            label="Capital in use"
            value={`₹${config.capital_to_use.toLocaleString("en-IN")}`}
            sub={`of ₹${config.total_capital.toLocaleString("en-IN")}`}
          />
          <StatCard
            label="Today's PnL"
            value={`₹${(dailyPnl?.realized_pnl ?? 0).toLocaleString("en-IN")}`}
            sub={`target ₹${config.daily_profit_target.toLocaleString("en-IN")} — ${
              dailyPnl?.target_hit ? "hit" : "not hit"
            }`}
            tone={(dailyPnl?.realized_pnl ?? 0) >= 0 ? "positive" : "negative"}
          />
          <StatCard
            label="Circuit breaker"
            value={dailyPnl?.circuit_breaker_triggered ? "TRIGGERED" : "Clear"}
            sub={`max daily loss ₹${config.max_daily_loss.toLocaleString("en-IN")}`}
            tone={dailyPnl?.circuit_breaker_triggered ? "negative" : "positive"}
          />
          <StatCard
            label="Trades today"
            value={String(dailyPnl?.trades_count ?? 0)}
            sub={`${config.max_concurrent_positions} max concurrent`}
          />
        </div>
      )}
    </div>
  );
}

function StatCard({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "positive" | "negative";
}) {
  const toneClass =
    tone === "positive" ? "text-emerald-600" : tone === "negative" ? "text-red-600" : "";
  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-4">
      <div className="text-xs text-neutral-500">{label}</div>
      <div className={`text-lg font-semibold ${toneClass}`}>{value}</div>
      {sub && <div className="text-xs text-neutral-500 mt-1">{sub}</div>}
    </div>
  );
}
