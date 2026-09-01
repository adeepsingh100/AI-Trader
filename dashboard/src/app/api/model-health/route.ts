import { NextResponse } from "next/server";
import { adminDb } from "@/lib/firebaseAdmin";
import { serializeTimestamps } from "@/lib/firestoreSerialize";

export async function GET() {
  try {
    const snap = await adminDb.collection("model_usage").orderBy("timestamp", "desc").limit(500).get();
    return NextResponse.json(serializeTimestamps(snap.docs.map((d) => ({ ...d.data(), id: d.id }))));
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
