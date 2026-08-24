import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";

// Backs trades-client.tsx's 20s live-price poll — kept as its own route
// (not folded into /api/trades) since it's a genuinely different table,
// filter, and cadence; combining them would re-fetch the 50-row trade
// list every 20s for nothing.
export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const symbolsParam = request.nextUrl.searchParams.get("symbols") ?? "";
  const symbols = symbolsParam.split(",").filter(Boolean);
  if (symbols.length === 0) return NextResponse.json([]);

  try {
    const res = await pool.query(
      "SELECT symbol, timestamp, features FROM opportunity_evaluations " +
        "WHERE mode = $1 AND symbol = ANY($2) ORDER BY timestamp DESC LIMIT 200",
      [mode, symbols]
    );
    return NextResponse.json(res.rows);
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
