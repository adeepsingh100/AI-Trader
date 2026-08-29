"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchJson } from "@/lib/api";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import { useStrategyType, StrategyTypeToggle } from "@/components/StrategyTypeToggle";
import { STATUS } from "@/lib/palette";
import type { Trade } from "@/lib/types";

const STATUS_COLOR: Record<Trade["status"], string> = {
  open: "#2a78d6",
  closed: "var(--text-muted)",
  flattened: STATUS.critical,
};

// How often to re-poll the latest close price for open positions. A push-
// based feed would update instantly, but needs infrastructure this app
// doesn't have — polling is the simpler, dependency-free way to get a
// "live" column without that setup.
const PRICE_POLL_MS = 20_000;

function timeframeMinutes(timeframe: string): number {
  const match = timeframe.match(/^(\d+)([mhd])$/);
  if (!match) return Infinity;
  const n = Number(match[1]);
  return match[2] === "m" ? n : match[2] === "h" ? n * 60 : n * 1440;
}

// The freshest close price across whatever timeframes this evaluation row's
// features cover — the shortest timeframe available (e.g. "1m" over "1h")
// is the closest proxy for "current price" a live UI column can get without
// a dedicated tick feed.
function latestClose(features: Record<string, { close?: number | null }> | null): number | null {
  if (!features) return null;
  const withClose = Object.entries(features).filter(([, v]) => typeof v?.close === "number");
  if (withClose.length === 0) return null;
  withClose.sort((a, b) => timeframeMinutes(a[0]) - timeframeMinutes(b[0]));
  return withClose[0][1].close as number;
}

export default function TradesClient() {
  const mode = useMode();
  const strategyType = useStrategyType();
  const [trades, setTrades] = useState<Trade[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentPrices, setCurrentPrices] = useState<Map<string, number>>(new Map());

  useEffect(() => {
    let cancelled = false;
    fetchJson<Trade[]>(`/api/trades?mode=${mode}&strategy_type=${strategyType}`)
      .then((data) => {
        if (cancelled) return;
        setError(null);
        setTrades(data);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setTrades(null);
      });
    return () => {
      cancelled = true;
    };
  }, [mode, strategyType]);

  const openSymbols = useMemo(
    () => Array.from(new Set((trades ?? []).filter((t) => t.status === "open").map((t) => t.symbol))).sort(),
    [trades]
  );
  const openSymbolsKey = openSymbols.join(",");

  useEffect(() => {
    // Nothing to poll when no trade is open — stale entries from a since-
    // closed symbol are harmless since the render below only reads this
    // map for rows whose status is still "open".
    if (openSymbols.length === 0) return;
    let cancelled = false;

    async function loadPrices() {
      const data = await fetchJson<{ symbol: string; features: Record<string, { close?: number | null }> }[]>(
        `/api/trades/prices?mode=${mode}&symbols=${openSymbols.join(",")}`
      ).catch(() => null);
      if (cancelled || !data) return;

      const bySymbol = new Map<string, number>();
      for (const row of data) {
        if (bySymbol.has(row.symbol)) continue; // rows are newest-first, keep the first hit per symbol
        const close = latestClose(row.features);
        if (close != null) bySymbol.set(row.symbol, close);
      }
      setCurrentPrices(bySymbol);
    }

    loadPrices();
    const interval = setInterval(loadPrices, PRICE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- openSymbolsKey is the stable primitive form of openSymbols
  }, [mode, openSymbolsKey]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Trades</h1>
        <div className="flex items-center gap-2">
          <StrategyTypeToggle />
          <ModeToggle />
        </div>
      </div>

      {error && (
        <p className="text-sm rounded-lg p-3" style={{ background: "#fdecea", color: "var(--status-critical)" }}>
          Query failed: {error}
        </p>
      )}
      {!error && trades === null && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
      {!error && trades?.length === 0 && (
        <p style={{ color: "var(--text-muted)" }}>No trades yet.</p>
      )}

      {trades && trades.length > 0 && (
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
                {[
                  "Symbol",
                  "Side",
                  "Qty",
                  "Entry",
                  "Stop-loss",
                  "Target",
                  "Current",
                  "Exit",
                  "Bought for",
                  "Sold for",
                  "PnL",
                  "Status",
                  "Opened",
                  "Closed",
                  "Reasoning",
                ].map((h) => (
                  <th key={h} className="px-3 py-2.5 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => (
                <tr
                  key={t.id}
                  className="align-top hover:bg-black/[0.02] transition-colors"
                  style={{ borderBottom: "1px solid var(--gridline)" }}
                >
                  <td className="px-3 py-2.5 font-medium">{t.symbol}</td>
                  <td className="px-3 py-2.5">
                    <span
                      className="text-xs font-medium px-1.5 py-0.5 rounded"
                      style={{
                        color: t.side === "buy" ? STATUS.good : "var(--text-secondary)",
                        background: t.side === "buy" ? "#eafaea" : "var(--page-plane)",
                      }}
                    >
                      {t.side}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 tabular-nums">{t.qty}</td>
                  <td className="px-3 py-2.5 tabular-nums">{t.entry_price}</td>
                  <td className="px-3 py-2.5 tabular-nums" style={{ color: STATUS.critical }}>
                    {t.stop_loss_price ?? "-"}
                  </td>
                  <td className="px-3 py-2.5 tabular-nums" style={{ color: STATUS.good }}>
                    {t.take_profit_price ?? "-"}
                  </td>
                  <td className="px-3 py-2.5 tabular-nums">
                    {(() => {
                      if (t.status !== "open") return "-";
                      const current = currentPrices.get(t.symbol);
                      if (current == null) return "-";
                      return (
                        <span style={{ color: current >= t.entry_price ? STATUS.good : STATUS.critical }}>
                          {current}
                        </span>
                      );
                    })()}
                  </td>
                  <td className="px-3 py-2.5 tabular-nums">{t.exit_price ?? "-"}</td>
                  <td className="px-3 py-2.5 tabular-nums whitespace-nowrap">
                    ₹{(t.qty * t.entry_price).toLocaleString("en-IN", { maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-3 py-2.5 tabular-nums whitespace-nowrap">
                    {t.exit_price != null
                      ? `₹${(t.qty * t.exit_price).toLocaleString("en-IN", { maximumFractionDigits: 2 })}`
                      : "-"}
                  </td>
                  <td
                    className="px-3 py-2.5 tabular-nums font-medium"
                    style={{
                      color: t.pnl == null ? "var(--text-primary)" : t.pnl >= 0 ? STATUS.good : STATUS.critical,
                    }}
                  >
                    {t.pnl ?? "-"}
                  </td>
                  <td className="px-3 py-2.5">
                    <span className="text-xs font-medium" style={{ color: STATUS_COLOR[t.status] }}>
                      {t.status}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 whitespace-nowrap" style={{ color: "var(--text-secondary)" }}>
                    {new Date(t.opened_at).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}
                  </td>
                  <td className="px-3 py-2.5 whitespace-nowrap" style={{ color: "var(--text-secondary)" }}>
                    {t.closed_at
                      ? new Date(t.closed_at).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })
                      : "-"}
                  </td>
                  <td className="px-3 py-2.5 max-w-xs" style={{ color: "var(--text-secondary)" }}>
                    {t.reasoning_text}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
