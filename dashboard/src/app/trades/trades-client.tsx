"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { useMode, ModeToggle } from "@/components/ModeToggle";
import type { Trade } from "@/lib/types";

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
        <h1 className="text-xl font-semibold">Trades</h1>
        <ModeToggle />
      </div>

      {error && (
        <p className="text-red-600 text-sm rounded border border-red-200 bg-red-50 p-3">
          Query failed: {error}
        </p>
      )}
      {!error && trades === null && <p className="text-neutral-500">Loading…</p>}
      {!error && trades?.length === 0 && <p className="text-neutral-500">No trades yet.</p>}

      {trades && trades.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-neutral-200 bg-white">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-neutral-500 border-b border-neutral-200">
                {["Symbol", "Side", "Qty", "Entry", "Exit", "PnL", "Status", "Opened", "Reasoning"].map(
                  (h) => (
                    <th key={h} className="px-3 py-2 font-medium">
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => (
                <tr key={t.id} className="border-b border-neutral-100 last:border-0 align-top">
                  <td className="px-3 py-2 font-medium">{t.symbol}</td>
                  <td className="px-3 py-2">{t.side}</td>
                  <td className="px-3 py-2">{t.qty}</td>
                  <td className="px-3 py-2">{t.entry_price}</td>
                  <td className="px-3 py-2">{t.exit_price ?? "-"}</td>
                  <td
                    className={
                      "px-3 py-2 " +
                      (t.pnl == null ? "" : t.pnl >= 0 ? "text-emerald-600" : "text-red-600")
                    }
                  >
                    {t.pnl ?? "-"}
                  </td>
                  <td className="px-3 py-2">{t.status}</td>
                  <td className="px-3 py-2 whitespace-nowrap">
                    {new Date(t.opened_at).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}
                  </td>
                  <td className="px-3 py-2 max-w-xs text-neutral-600">{t.reasoning_text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
