"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import { STATUS } from "@/lib/palette";
import type { Trade } from "@/lib/types";

const STATUS_COLOR: Record<Trade["status"], string> = {
  open: "#2a78d6",
  closed: "var(--text-muted)",
  flattened: STATUS.critical,
};

export default function TradesClient() {
  const mode = useMode();
  const [trades, setTrades] = useState<Trade[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    supabase
      .from("trades")
      .select("*")
      .eq("mode", mode)
      .order("opened_at", { ascending: false })
      .limit(50)
      .then(({ data, error }) => {
        if (cancelled) return;
        setError(error ? error.message : null);
        setTrades(error ? null : (data ?? []));
      });
    return () => {
      cancelled = true;
    };
  }, [mode]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Trades</h1>
        <ModeToggle />
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
                  "Exit",
                  "Bought for",
                  "Sold for",
                  "PnL",
                  "Status",
                  "Opened",
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
