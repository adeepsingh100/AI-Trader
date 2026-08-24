import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";

  try {
    const res = await pool.query(
      "SELECT * FROM trades WHERE mode = $1 ORDER BY opened_at DESC LIMIT 50",
      [mode]
    );
    return NextResponse.json(res.rows);
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
