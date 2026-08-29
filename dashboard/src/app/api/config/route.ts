import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";
import { COOKIE_NAME, isValidSession } from "@/lib/configAuth";

const ALLOWED_FIELDS = new Set([
  "total_capital",
  "capital_to_use",
  "daily_profit_target",
  "max_daily_loss",
  "position_size_pct",
  "max_concurrent_positions",
  "paused",
]);

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";
  try {
    const res = await pool.query(
      "SELECT * FROM capital_config WHERE mode = $1 AND strategy_type = $2",
      [mode, strategyType]
    );
    return NextResponse.json(res.rows[0] ?? null);
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}

export async function PATCH(request: NextRequest) {
  const cookieStore = await cookies();
  if (!isValidSession(cookieStore.get(COOKIE_NAME)?.value)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";
  const body = await request.json().catch(() => ({}));
  const fields = Object.keys(body).filter((k) => ALLOWED_FIELDS.has(k));
  if (fields.length === 0) {
    return NextResponse.json({ error: "no valid fields in body" }, { status: 400 });
  }

  const setClause = fields.map((f, i) => `${f} = $${i + 1}`).join(", ");
  const values = fields.map((f) => body[f]);
  try {
    await pool.query(
      `UPDATE capital_config SET ${setClause} WHERE mode = $${fields.length + 1} AND strategy_type = $${fields.length + 2}`,
      [...values, mode, strategyType]
    );
    return NextResponse.json({ ok: true });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
