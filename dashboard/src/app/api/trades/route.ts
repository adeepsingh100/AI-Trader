import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";

  try {
    const res = await pool.query(
      "SELECT t.* FROM trades t JOIN strategy_versions sv ON t.version_id = sv.id " +
        "WHERE t.mode = $1 AND sv.strategy_type = $2 ORDER BY t.opened_at DESC LIMIT 50",
      [mode, strategyType]
    );
    return NextResponse.json(res.rows);
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
