"use client";

import { useEffect, useState } from "react";
import { fetchJson } from "@/lib/api";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import { StatCard } from "@/components/StatCard";
import type { CapitalConfig, DailyPnl } from "@/lib/types";

interface OverviewData {
  config: CapitalConfig | null;
  dailyPnl: DailyPnl | null;
  openTrades: { qty: number; entry_price: number }[];
}

export default function OverviewClient() {
  const mode = useMode();
  const [config, setConfig] = useState<CapitalConfig | null | undefined>(undefined);
  const [dailyPnl, setDailyPnl] = useState<DailyPnl | null>(null);
  const [capitalInUse, setCapitalInUse] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await fetchJson<OverviewData>(`/api/overview?mode=${mode}`);
        if (cancelled) return;
        setError(null);
        setConfig(data.config);
        setDailyPnl(data.dailyPnl);
        // Committed capital = sum(qty * entry_price) over open positions —
        // same formula risk_manager.py's committed_capital() gates sizing against.
        setCapitalInUse(data.openTrades.reduce((sum, t) => sum + t.qty * t.entry_price, 0));
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setConfig(undefined);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [mode]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <ModeToggle />
      </div>

      {error && (
        <p className="text-sm rounded-lg p-3" style={{ background: "#fdecea", color: "var(--status-critical)" }}>
          Query failed: {error}
        </p>
      )}
      {!error && config === undefined && (
        <p style={{ color: "var(--text-muted)" }}>Loading…</p>
      )}
      {!error && config === null && (
        <p style={{ color: "var(--text-muted)" }}>
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
            tone={config.paused ? "critical" : "good"}
          />
          <StatCard
            label="Capital in use"
            value={`₹${capitalInUse.toLocaleString("en-IN")}`}
            sub={`₹${Math.max(config.capital_to_use - capitalInUse, 0).toLocaleString("en-IN")} left of ₹${config.capital_to_use.toLocaleString("en-IN")} limit`}
            tone={capitalInUse > config.capital_to_use ? "critical" : "neutral"}
          />
          <StatCard
            label="Today's PnL"
            value={`₹${(dailyPnl?.realized_pnl ?? 0).toLocaleString("en-IN")}`}
            sub={`target ₹${config.daily_profit_target.toLocaleString("en-IN")} — ${
              dailyPnl?.target_hit ? "hit" : "not hit"
            }`}
            tone={(dailyPnl?.realized_pnl ?? 0) >= 0 ? "good" : "critical"}
          />
          <StatCard
            label="Circuit breaker"
            value={dailyPnl?.circuit_breaker_triggered ? "TRIGGERED" : "Clear"}
            sub={`max daily loss ₹${config.max_daily_loss.toLocaleString("en-IN")}`}
            tone={dailyPnl?.circuit_breaker_triggered ? "critical" : "good"}
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
