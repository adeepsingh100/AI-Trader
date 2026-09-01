import { NextRequest, NextResponse } from "next/server";
import { adminDb } from "@/lib/firebaseAdmin";
import { serializeTimestamps } from "@/lib/firestoreSerialize";

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
    const snap = await adminDb.collection("capital_config").doc(`${mode}_${strategyType}`).get();
    return NextResponse.json(snap.exists ? serializeTimestamps({ ...snap.data(), id: snap.id }) : null);
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}

// No password/session check here anymore — proxy.ts already gates every
// route behind Firebase Auth for the whole dashboard, so a second
// per-route check would be redundant (was the old isValidSession check
// against CONFIG_EDIT_PASSWORD, now deleted).
export async function PATCH(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";
  const body = await request.json().catch(() => ({}));
  const fields = Object.fromEntries(Object.entries(body).filter(([k]) => ALLOWED_FIELDS.has(k)));
  if (Object.keys(fields).length === 0) {
    return NextResponse.json({ error: "no valid fields in body" }, { status: 400 });
  }

  try {
    await adminDb.collection("capital_config").doc(`${mode}_${strategyType}`).set(fields, { merge: true });
    return NextResponse.json({ ok: true });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
