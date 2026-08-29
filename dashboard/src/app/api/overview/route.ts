import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";
import { todayIst } from "@/lib/date";

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";

  try {
    const [configRes, pnlRes, tradesRes] = await Promise.all([
      pool.query("SELECT * FROM capital_config WHERE mode = $1 AND strategy_type = $2", [mode, strategyType]),
      pool.query(
        "SELECT * FROM daily_pnl WHERE mode = $1 AND strategy_type = $2 AND date = $3",
        [mode, strategyType, todayIst()]
      ),
      pool.query(
        "SELECT t.qty, t.entry_price FROM trades t JOIN strategy_versions sv ON t.version_id = sv.id " +
          "WHERE t.mode = $1 AND t.status = 'open' AND sv.strategy_type = $2",
        [mode, strategyType]
      ),
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
