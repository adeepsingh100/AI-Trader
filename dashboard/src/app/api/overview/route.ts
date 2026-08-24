import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";
import { todayIst } from "@/lib/date";

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";

  try {
    const [configRes, pnlRes, tradesRes] = await Promise.all([
      pool.query("SELECT * FROM capital_config WHERE mode = $1", [mode]),
      pool.query("SELECT * FROM daily_pnl WHERE mode = $1 AND date = $2", [mode, todayIst()]),
      pool.query("SELECT qty, entry_price FROM trades WHERE mode = $1 AND status = 'open'", [mode]),
    ]);
    return NextResponse.json({
      config: configRes.rows[0] ?? null,
      dailyPnl: pnlRes.rows[0] ?? null,
      openTrades: tradesRes.rows,
    });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
