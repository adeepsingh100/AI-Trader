import { NextRequest, NextResponse } from "next/server";
import { adminDb } from "@/lib/firebaseAdmin";
import { serializeTimestamps } from "@/lib/firestoreSerialize";
import { Timestamp } from "firebase-admin/firestore";

// LEARNING_HISTORY_WINDOW_DAYS default (src/config.py) — same window
// rejection_breakdown() uses when no explicit `since` is passed.
const HISTORY_WINDOW_DAYS = 180;

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const strategyType = request.nextUrl.searchParams.get("strategy_type") ?? "default";
  const since = Timestamp.fromMillis(Date.now() - HISTORY_WINDOW_DAYS * 24 * 60 * 60 * 1000);

  try {
    const [statsSnap, evalsSnap, recsSnap, simsSnap, versionsSnap, tradesSnap, latestVersionSnap, featuresSnap] =
      await Promise.all([
        adminDb
          .collection("learning_statistics")
          .where("mode", "==", mode)
          .where("strategy_type", "==", strategyType)
          .get(),
        adminDb
          .collection("opportunity_evaluations")
          .where("mode", "==", mode)
          .where("strategy_type", "==", strategyType)
          .where("timestamp", ">=", since)
          .limit(5000)
          .get(),
        adminDb
          .collection("recommendations")
          .where("mode", "==", mode)
          .where("strategy_type", "==", strategyType)
          .orderBy("created_at", "desc")
          .get(),
        adminDb
          .collection("strategy_simulations")
          .where("mode", "==", mode)
          .where("strategy_type", "==", strategyType)
          .orderBy("created_at", "desc")
          .get(),
        adminDb
          .collection("adaptive_strategy_versions")
          .where("mode", "==", mode)
          .where("strategy_type", "==", strategyType)
          .orderBy("created_at", "desc")
          .get(),
        adminDb
          .collection("trades")
          .where("mode", "==", mode)
          .where("strategy_type", "==", strategyType)
          .where("status", "==", "closed")
          .where("closed_at", ">=", since)
          .get(),
        adminDb
          .collection("strategy_versions")
          .where("strategy_type", "==", strategyType)
          .orderBy("version_number", "desc")
          .limit(1)
          .get(),
        adminDb
          .collection("feature_importance")
          .where("mode", "==", mode)
          .where("strategy_type", "==", strategyType)
          .get(),
      ]);

    return NextResponse.json(
      serializeTimestamps({
        learningStats: statsSnap.docs.map((d) => d.data()),
        evaluations: evalsSnap.docs.map((d) => d.data()),
        featureNames: featuresSnap.docs
          .map((d) => d.data())
          .filter((f) => f.timeframe !== "blended")
          .map((f) => f.feature_name),
        recommendations: recsSnap.docs.map((d) => d.data()),
        simulations: simsSnap.docs.map((d) => ({ ...d.data(), id: d.id })),
        versions: versionsSnap.docs.map((d) => d.data()),
        closedTrades: tradesSnap.docs.map((d) => d.data()),
        promotionEligible: latestVersionSnap.docs[0]?.data()?.promotion_eligible ?? false,
      })
    );
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
