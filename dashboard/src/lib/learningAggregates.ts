import type { HoldEvaluation, LearningStage, LearningStatistic, LearningStatus } from "@/lib/types";

export interface RejectionCount {
  reason: string;
  count: number;
  pctOfRejections: number;
}

// Mirrors src/learning/rejection_analysis.py::rejection_breakdown +
// _rejection_label — risk_manager_result is the more specific reason when
// present, reason otherwise, "unknown" if neither — grouped and sorted
// desc by count, so the dashboard and the HTML report never disagree.
export function rejectionBreakdown(rows: HoldEvaluation[]): RejectionCount[] {
  if (rows.length === 0) return [];

  const counts = new Map<string, number>();
  for (const row of rows) {
    const label = row.risk_manager_result || row.reason || "unknown";
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  const total = rows.length;
  return Array.from(counts.entries())
    .map(([reason, count]) => ({ reason, count, pctOfRejections: (count / total) * 100 }))
    .sort((a, b) => b.count - a.count);
}

export interface WeaknessBucket {
  value: string;
  expectancy: number;
  trades_count: number;
}

// Mirrors src/learning/weakness_detection.py::identify_weaknesses's
// worst_by_dimension — the only half reports.py's own _weakness_rows
// renders, so this matches the reference being mirrored (best_by_dimension
// and the indicator extremes aren't surfaced there either).
export function worstBucketByDimension(
  rows: LearningStatistic[],
  minSampleSize = 25
): Record<string, WeaknessBucket> {
  const byDimension = new Map<string, LearningStatistic[]>();
  for (const row of rows) {
    if ((row.trades_count || 0) < minSampleSize || row.expectancy == null) continue;
    const bucket = byDimension.get(row.dimension_type) ?? [];
    bucket.push(row);
    byDimension.set(row.dimension_type, bucket);
  }

  const worst: Record<string, WeaknessBucket> = {};
  for (const [dimensionType, eligible] of byDimension) {
    const min = eligible.reduce((a, b) => ((b.expectancy as number) < (a.expectancy as number) ? b : a));
    worst[dimensionType] = {
      value: min.dimension_value,
      expectancy: min.expectancy as number,
      trades_count: min.trades_count,
    };
  }
  return worst;
}

// Progressive Learning Stages boundaries — mirrors
// LEARNING_STAGE_OBSERVATION/HYPOTHESIS/SIMULATION/VALIDATION_MIN_TRADES
// (src/config.py) and src/learning/learning_status.py::compute_learning_status.
const OBSERVATION_MIN_TRADES = 25;
const HYPOTHESIS_MIN_TRADES = 100;
const SIMULATION_MIN_TRADES = 250;
const VALIDATION_MIN_TRADES = 500;

const STAGE_ACTIVITY: Record<LearningStage, string> = {
  BOOTSTRAP: "Collecting trade data, rejection reasons, and feature distributions only. No analysis yet.",
  OBSERVATION: "Analyzing rejection reasons, feature distributions, and weakness patterns. No strategy changes yet.",
  HYPOTHESIS: "Generating hypotheses (weight/threshold/exit-parameter recommendations) from observed weaknesses. No candidate strategies yet.",
  SIMULATION: "Testing hypotheses via backtest and walk-forward simulation. Candidates are validated but not yet created.",
  VALIDATION: "Full validation active — passing simulations create candidate strategies, pending human approval for promotion.",
};

function stageFor(tradesCollected: number): [LearningStage, LearningStage | null, number | null] {
  if (tradesCollected < OBSERVATION_MIN_TRADES) return ["BOOTSTRAP", "OBSERVATION", OBSERVATION_MIN_TRADES];
  if (tradesCollected < HYPOTHESIS_MIN_TRADES) return ["OBSERVATION", "HYPOTHESIS", HYPOTHESIS_MIN_TRADES];
  if (tradesCollected < SIMULATION_MIN_TRADES) return ["HYPOTHESIS", "SIMULATION", SIMULATION_MIN_TRADES];
  if (tradesCollected < VALIDATION_MIN_TRADES) return ["SIMULATION", "VALIDATION", VALIDATION_MIN_TRADES];
  return ["VALIDATION", null, null];
}

// Mirrors src/learning/learning_status.py::compute_learning_status — same
// stage boundaries, same field set, so the dashboard and the HTML report
// never disagree about what stage a mode is in.
export function computeLearningStage(
  closedTrades: { pnl: number | null }[],
  rejectedTrades: number,
  recommendationsCount: number,
  simulationsCount: number,
  candidatesCount: number,
  promotionEligible: boolean
): LearningStatus {
  const tradesCollected = closedTrades.length;
  const winningTrades = closedTrades.filter((t) => (t.pnl ?? 0) > 0).length;
  const losingTrades = tradesCollected - winningTrades;
  const [stage, nextStage, nextMin] = stageFor(tradesCollected);
  const tradesToNextStage = nextMin != null ? Math.max(0, nextMin - tradesCollected) : 0;
  const reason =
    nextStage == null
      ? `Full validation stage reached (${tradesCollected} trades collected).`
      : `${tradesToNextStage} more closed trade(s) needed to reach ${nextStage} (requires ${nextMin}).`;

  return {
    stage,
    tradesCollected,
    rejectedTrades,
    winningTrades,
    losingTrades,
    dataSufficiencyPct: Math.min(100, (tradesCollected / VALIDATION_MIN_TRADES) * 100),
    recommendationsCount,
    simulationsCount,
    candidatesCount,
    promotionEligible,
    nextStage,
    tradesToNextStage,
    currentActivity: STAGE_ACTIVITY[stage],
    reason,
  };
}
