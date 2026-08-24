import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";

// LEARNING_HISTORY_WINDOW_DAYS default (src/config.py) — same window
// rejection_breakdown() uses when no explicit `since` is passed.
const HISTORY_WINDOW_DAYS = 180;

export async function GET(request: NextRequest) {
  const mode = request.nextUrl.searchParams.get("mode") ?? "paper";
  const since = new Date(Date.now() - HISTORY_WINDOW_DAYS * 24 * 60 * 60 * 1000).toISOString();

  try {
    const [stats, evals, recs, sims, versions, trades, strategyVersions, features] = await Promise.all([
      pool.query(
        "SELECT dimension_type, dimension_value, expectancy, trades_count FROM learning_statistics WHERE mode = $1",
        [mode]
      ),
      pool.query(
        "SELECT symbol, market_regime, timestamp, final_decision, llm_decision, reason, risk_manager_result " +
          "FROM opportunity_evaluations WHERE mode = $1 AND timestamp >= $2 LIMIT 5000",
        [mode, since]
      ),
      pool.query(
        "SELECT category, metric_name, current_value, recommended_value, confidence, sample_size, rationale, status, created_at " +
          "FROM recommendations WHERE mode = $1 ORDER BY created_at DESC",
        [mode]
      ),
      pool.query(
        "SELECT id, created_at, passed, p_value, research_note FROM strategy_simulations WHERE mode = $1 " +
          "ORDER BY created_at DESC",
        [mode]
      ),
      pool.query(
        "SELECT version_number, status, fitness_score, notes, created_at, source_simulation_id " +
          "FROM adaptive_strategy_versions WHERE mode = $1 ORDER BY created_at DESC",
        [mode]
      ),
      pool.query(
        "SELECT pnl FROM trades WHERE mode = $1 AND status = 'closed' AND closed_at >= $2",
        [mode, since]
      ),
      pool.query(
        "SELECT promotion_eligible FROM strategy_versions ORDER BY version_number DESC LIMIT 1"
      ),
      pool.query("SELECT feature_name, timeframe FROM feature_importance WHERE mode = $1", [mode]),
    ]);

    return NextResponse.json({
      learningStats: stats.rows,
      evaluations: evals.rows,
      featureNames: features.rows.filter((f) => f.timeframe !== "blended").map((f) => f.feature_name),
      recommendations: recs.rows,
      simulations: sims.rows,
      versions: versions.rows,
      closedTrades: trades.rows,
      promotionEligible: strategyVersions.rows[0]?.promotion_eligible ?? false,
    });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
