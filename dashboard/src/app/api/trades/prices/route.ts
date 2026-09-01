import { NextRequest, NextResponse } from "next/server";
import { adminDb } from "@/lib/firebaseAdmin";
import { serializeTimestamps } from "@/lib/firestoreSerialize";

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
    // Firestore's "in" operator caps at 30 values, unlike Postgres's
    // unbounded = ANY($2) — chunk and merge.
    const chunks = [];
    for (let i = 0; i < symbols.length; i += 30) chunks.push(symbols.slice(i, i + 30));

    const results = await Promise.all(
      chunks.map((chunk) =>
        adminDb
          .collection("opportunity_evaluations")
          .where("mode", "==", mode)
          .where("symbol", "in", chunk)
          .orderBy("timestamp", "desc")
          .limit(200)
          .get()
      )
    );
    const rows = results
      .flatMap((snap) => snap.docs.map((d) => d.data()))
      .sort((a, b) => (b.timestamp?.toMillis?.() ?? 0) - (a.timestamp?.toMillis?.() ?? 0))
      .slice(0, 200);
    return NextResponse.json(serializeTimestamps(rows));
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
