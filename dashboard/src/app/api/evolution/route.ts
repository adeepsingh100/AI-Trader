import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";

  try {
    const [versionsRes, pnlRes] = await Promise.all([
      // Not mode-filtered — strategy_versions spans both modes, matches
      // the version-history table shown today. IS strategy_type-filtered —
      // each type is its own independent lineage, blending them would
      // interleave unrelated version histories.
      pool.query(
        "SELECT * FROM strategy_versions WHERE strategy_type = $1 ORDER BY version_number DESC",
        [strategyType]
      ),
      pool.query(
        "SELECT * FROM daily_pnl WHERE mode = $1 AND strategy_type = $2 ORDER BY date ASC",
        [mode, strategyType]
      ),
    ]);
    return NextResponse.json({ versions: versionsRes.rows, dailyPnl: pnlRes.rows });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
