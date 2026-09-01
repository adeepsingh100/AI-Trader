import { NextRequest, NextResponse } from "next/server";
import { adminDb } from "@/lib/firebaseAdmin";
import { serializeTimestamps } from "@/lib/firestoreSerialize";

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";

  try {
    const [versionsSnap, pnlSnap] = await Promise.all([
      // Not mode-filtered — strategy_versions spans both modes, matches
      // the version-history table shown today. IS strategy_type-filtered —
      // each type is its own independent lineage, blending them would
      // interleave unrelated version histories.
      adminDb
        .collection("strategy_versions")
        .where("strategy_type", "==", strategyType)
        .orderBy("version_number", "desc")
        .get(),
      adminDb
        .collection("daily_pnl")
        .where("mode", "==", mode)
        .where("strategy_type", "==", strategyType)
        .orderBy("date", "asc")
        .get(),
    ]);
    return NextResponse.json(
      serializeTimestamps({
        versions: versionsSnap.docs.map((d) => ({ ...d.data(), id: d.id })),
        dailyPnl: pnlSnap.docs.map((d) => d.data()),
      })
    );
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
