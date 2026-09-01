import { NextRequest, NextResponse } from "next/server";
import { adminDb } from "@/lib/firebaseAdmin";
import { serializeTimestamps } from "@/lib/firestoreSerialize";

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";

  try {
    const snap = await adminDb
      .collection("trades")
      .where("mode", "==", mode)
      .where("strategy_type", "==", strategyType)
      .orderBy("opened_at", "desc")
      .limit(50)
      .get();
    return NextResponse.json(serializeTimestamps(snap.docs.map((d) => ({ ...d.data(), id: d.id }))));
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
