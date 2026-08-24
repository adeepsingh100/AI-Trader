import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";

  try {
    const [versionsRes, pnlRes] = await Promise.all([
      // Not mode-filtered — strategy_versions spans both modes, matches
      // the version-history table shown today.
      pool.query("SELECT * FROM strategy_versions ORDER BY version_number DESC"),
      pool.query("SELECT * FROM daily_pnl WHERE mode = $1 ORDER BY date ASC", [mode]),
    ]);
    return NextResponse.json({ versions: versionsRes.rows, dailyPnl: pnlRes.rows });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
