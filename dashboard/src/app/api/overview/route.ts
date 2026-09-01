import { NextRequest, NextResponse } from "next/server";
import { adminDb } from "@/lib/firebaseAdmin";
import { serializeTimestamps } from "@/lib/firestoreSerialize";
import { todayIst } from "@/lib/date";

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";

  try {
    const [configSnap, pnlSnap, tradesSnap] = await Promise.all([
      adminDb.collection("capital_config").doc(`${mode}_${strategyType}`).get(),
      adminDb.collection("daily_pnl").doc(`${todayIst()}_${mode}_${strategyType}`).get(),
      adminDb
        .collection("trades")
        .where("mode", "==", mode)
        .where("status", "==", "open")
        .where("strategy_type", "==", strategyType)
        .get(),
    ]);
    return NextResponse.json(
      serializeTimestamps({
        config: configSnap.exists ? configSnap.data() : null,
        dailyPnl: pnlSnap.exists ? pnlSnap.data() : null,
        openTrades: tradesSnap.docs.map((d) => d.data()),
      })
    );
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
