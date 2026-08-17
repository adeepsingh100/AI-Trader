import type { Evidence, LearningStage, LearningStatistic, LearningStatus, OpportunityEvaluationRow } from "@/lib/types";

export interface RejectionCount {
  reason: string;
  count: number;
  pctOfRejections: number;
}

// Mirrors src/learning/rejection_analysis.py::rejection_breakdown +
// _rejection_label — risk_manager_result is the more specific reason when
// present, reason otherwise, "unknown" if neither — grouped and sorted
// desc by count, so the dashboard and the HTML report never disagree.
export function rejectionBreakdown(rows: { reason: string | null; risk_manager_result: string | null }[]): RejectionCount[] {
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

// --- Evidence Engine mirror (src/learning/evidence_engine.py) ---------------

// classify_market_regime's exhaustive label set (opportunity_scorer.py) —
// the real total, not a guess.
const ALL_MARKET_REGIMES = new Set([
  "sideways", "high_volatility", "strong_bull", "weak_bull", "strong_bear", "weak_bear",
]);
const ACCEPTED_DECISIONS = new Set(["buy", "sell"]);
// Per-bucket observation floor (RECOMMENDATION_MIN_SAMPLE_SIZE default,
// src/config.py) — a symbol needs to have been SEEN this many times
// before "rarely qualifying" means anything.
const SYMBOL_SEEN_FLOOR = 20;
// The 8 dimension_types learning_statistics tracks (src/learning/statistics.py::_DIMENSION_TYPES).
const TRACKED_DIMENSION_TYPES = 8;
// len(feature_engine.FEATURE_KEYS) — a Python-side code fact, kept in sync
// by hand (no shared runtime between Python and TS, same as every other
// mirrored constant in this file). A few keys there are booleans/
// categoricals never correlated, so 100% coverage is never literally
// reachable, same honest cap the Python side documents.
const TOTAL_FEATURE_KEYS = 26;

function istHour(timestamp: string): number {
  const date = new Date(timestamp);
  const istMinutes = (date.getUTCHours() * 60 + date.getUTCMinutes() + 5 * 60 + 30) % (24 * 60);
  return Math.floor(istMinutes / 60);
}

// Mirrors src/learning/evidence_engine.py::EvidenceEngine.collect — same
// dimensions, same denominators, so the dashboard and the nightly report
// never disagree about how much evidence a mode has collected.
export function collectEvidence(
  closedTrades: { pnl: number | null }[],
  evaluations: OpportunityEvaluationRow[],
  featureNamesSeen: Set<string>,
  dimensionTypesSeen: Set<string>
): Evidence {
  const winningTrades = closedTrades.filter((t) => (t.pnl ?? 0) > 0).length;

  const rejected = evaluations.filter((e) => e.final_decision === "hold");
  const acceptedSymbols = new Set(
    evaluations.filter((e) => e.final_decision && ACCEPTED_DECISIONS.has(e.final_decision)).map((e) => e.symbol)
  );
  const acceptedRegimes = new Set(
    evaluations
      .filter((e) => e.final_decision && ACCEPTED_DECISIONS.has(e.final_decision) && e.market_regime)
      .map((e) => e.market_regime as string)
  );
  const symbolsSeen = new Set(evaluations.filter((e) => e.symbol).map((e) => e.symbol as string));
  const regimesSeen = new Set(evaluations.filter((e) => e.market_regime).map((e) => e.market_regime as string));
  const hoursSeen = new Set(evaluations.filter((e) => e.timestamp).map((e) => istHour(e.timestamp as string)));
  const reachedLlm = evaluations.filter((e) => e.llm_decision != null).length;

  const symbolSeenCounts = new Map<string, number>();
  const symbolRejectCounts = new Map<string, number>();
  for (const e of evaluations) {
    if (!e.symbol) continue;
    symbolSeenCounts.set(e.symbol, (symbolSeenCounts.get(e.symbol) ?? 0) + 1);
  }
  for (const e of rejected) {
    if (!e.symbol) continue;
    symbolRejectCounts.set(e.symbol, (symbolRejectCounts.get(e.symbol) ?? 0) + 1);
  }
  const symbolsRarelyQualifying = Array.from(symbolSeenCounts.entries())
    .filter(([symbol, count]) => count >= SYMBOL_SEEN_FLOOR && !acceptedSymbols.has(symbol))
    .map(([symbol, seen]) => ({
      symbol,
      seen,
      rejectRatePct: ((symbolRejectCounts.get(symbol) ?? 0) / seen) * 100,
    }))
    .sort((a, b) => b.seen - a.seen)
    .slice(0, 10);

  const regimesWithNoCandidates = Array.from(regimesSeen)
    .filter((r) => !acceptedRegimes.has(r))
    .sort();

  const marketRegimesCovered = Array.from(regimesSeen).filter((r) => ALL_MARKET_REGIMES.has(r)).length;

  return {
    closedTrades: closedTrades.length,
    winningTrades,
    losingTrades: closedTrades.length - winningTrades,
    rejectedOpportunities: rejected.length,
    candidateOpportunities: evaluations.length,
    symbolsCovered: symbolsSeen.size,
    marketRegimesCovered,
    tradingHoursCovered: hoursSeen.size,
    featureCoveragePct: clamp((featureNamesSeen.size / TOTAL_FEATURE_KEYS) * 100, 0, 100),
    confidenceCoveragePct: evaluations.length ? clamp((reachedLlm / evaluations.length) * 100, 0, 100) : 0,
    learningCoveragePct: clamp((dimensionTypesSeen.size / TRACKED_DIMENSION_TYPES) * 100, 0, 100),
    symbolsRarelyQualifying,
    regimesWithNoCandidates,
  };
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

// Evidence Readiness blend weights — mirrors EVIDENCE_WEIGHT_* defaults
// (src/config.py) and evidence_engine.py::compute_evidence_readiness.
const EVIDENCE_WEIGHTS = {
  tradeCoverage: 0.25,
  marketCoverage: 0.15,
  symbolCoverage: 0.15,
  featureCoverage: 0.15,
  sessionCoverage: 0.10,
  confidenceCoverage: 0.10,
  rejectionEvidence: 0.10,
};
const SYMBOL_COVERAGE_TARGET = 20;
const REJECTED_COVERAGE_TARGET = 500;
const VALIDATION_MIN_TRADES = 500;

export interface EvidenceReadiness {
  evidenceReadinessPct: number;
  components: Record<keyof typeof EVIDENCE_WEIGHTS, number>;
}

// Mirrors evidence_engine.py::compute_evidence_readiness — same weighted-
// average-over-available blend opportunity_scorer.weighted_average uses
// on the Python side (every component here is always a concrete number,
// never missing, so this is a plain weighted sum).
export function computeEvidenceReadiness(evidence: Evidence): EvidenceReadiness {
  const components = {
    tradeCoverage: clamp((evidence.closedTrades / VALIDATION_MIN_TRADES) * 100, 0, 100),
    marketCoverage: clamp((evidence.marketRegimesCovered / ALL_MARKET_REGIMES.size) * 100, 0, 100),
    symbolCoverage: clamp((evidence.symbolsCovered / SYMBOL_COVERAGE_TARGET) * 100, 0, 100),
    featureCoverage: evidence.featureCoveragePct,
    sessionCoverage: clamp((evidence.tradingHoursCovered / 24) * 100, 0, 100),
    confidenceCoverage: evidence.confidenceCoveragePct,
    rejectionEvidence: clamp((evidence.rejectedOpportunities / REJECTED_COVERAGE_TARGET) * 100, 0, 100),
  };
  let total = 0;
  let totalWeight = 0;
  for (const key of Object.keys(EVIDENCE_WEIGHTS) as (keyof typeof EVIDENCE_WEIGHTS)[]) {
    total += components[key] * EVIDENCE_WEIGHTS[key];
    totalWeight += EVIDENCE_WEIGHTS[key];
  }
  return { evidenceReadinessPct: totalWeight ? total / totalWeight : 0, components };
}

// --- Stage computation (src/learning/learning_status.py) --------------------
// Load-bearing decision, same as the Python side: BOOTSTRAP->OBSERVATION is
// evidence-driven (any ONE dimension clearing its bar unlocks it — rejection
// analysis and coverage reporting don't need a closed trade). HYPOTHESIS/
// SIMULATION/VALIDATION stay trade-count gated — those are statistical
// procedures on trade OUTCOMES that no amount of symbol/regime coverage
// substitutes for.
const OBSERVATION_MIN_TRADES = 25;
const HYPOTHESIS_MIN_TRADES = 100;
const SIMULATION_MIN_TRADES = 250;
const HOURS_OBSERVATION_MIN = 25;
const REGIMES_OBSERVATION_MIN = 6;
const FEATURE_COVERAGE_OBSERVATION_MIN_PCT = 80;
const READINESS_OBSERVATION_MIN_PCT = 40;

const STAGE_ACTIVITY: Record<LearningStage, string> = {
  BOOTSTRAP: "Collecting trade data, rejection reasons, and feature distributions only. No analysis yet.",
  OBSERVATION: "Analyzing rejection reasons, feature distributions, and weakness patterns. No strategy changes yet.",
  HYPOTHESIS: "Generating hypotheses (weight/threshold/exit-parameter recommendations) from observed weaknesses. No candidate strategies yet.",
  SIMULATION: "Testing hypotheses via backtest and walk-forward simulation. Candidates are validated but not yet created.",
  VALIDATION: "Full validation active — passing simulations create candidate strategies, pending human approval for promotion.",
};

function observationReady(evidence: Evidence, evidenceReadinessPct: number): boolean {
  return (
    evidence.closedTrades >= OBSERVATION_MIN_TRADES ||
    evidence.rejectedOpportunities >= REJECTED_COVERAGE_TARGET ||
    evidence.tradingHoursCovered >= HOURS_OBSERVATION_MIN ||
    evidence.symbolsCovered >= SYMBOL_COVERAGE_TARGET ||
    evidence.marketRegimesCovered >= REGIMES_OBSERVATION_MIN ||
    evidence.featureCoveragePct >= FEATURE_COVERAGE_OBSERVATION_MIN_PCT ||
    evidenceReadinessPct >= READINESS_OBSERVATION_MIN_PCT
  );
}

function bootstrapGaps(evidence: Evidence, evidenceReadinessPct: number): string[] {
  const gaps: string[] = [];
  const tradesGap = OBSERVATION_MIN_TRADES - evidence.closedTrades;
  if (tradesGap > 0) gaps.push(`${tradesGap} more closed trades`);
  const rejectedGap = REJECTED_COVERAGE_TARGET - evidence.rejectedOpportunities;
  if (rejectedGap > 0) gaps.push(`${rejectedGap} more rejected candidates`);
  const hoursGap = HOURS_OBSERVATION_MIN - evidence.tradingHoursCovered;
  if (hoursGap > 0) gaps.push(`${hoursGap} more trading hours covered`);
  const symbolsGap = SYMBOL_COVERAGE_TARGET - evidence.symbolsCovered;
  if (symbolsGap > 0) gaps.push(`${symbolsGap} more symbols covered`);
  const regimesGap = REGIMES_OBSERVATION_MIN - evidence.marketRegimesCovered;
  if (regimesGap > 0) gaps.push(`${regimesGap} more market regimes covered`);
  const featureGap = FEATURE_COVERAGE_OBSERVATION_MIN_PCT - evidence.featureCoveragePct;
  if (featureGap > 0) gaps.push(`${featureGap.toFixed(0)}% more feature coverage`);
  const readinessGap = READINESS_OBSERVATION_MIN_PCT - evidenceReadinessPct;
  if (readinessGap > 0) gaps.push(`${readinessGap.toFixed(0)}% more evidence readiness`);
  return gaps;
}

function stageFor(evidence: Evidence, evidenceReadinessPct: number): [LearningStage, LearningStage | null, number | null] {
  if (!observationReady(evidence, evidenceReadinessPct)) return ["BOOTSTRAP", "OBSERVATION", null];
  const trades = evidence.closedTrades;
  if (trades < HYPOTHESIS_MIN_TRADES) return ["OBSERVATION", "HYPOTHESIS", HYPOTHESIS_MIN_TRADES];
  if (trades < SIMULATION_MIN_TRADES) return ["HYPOTHESIS", "SIMULATION", SIMULATION_MIN_TRADES];
  if (trades < VALIDATION_MIN_TRADES) return ["SIMULATION", "VALIDATION", VALIDATION_MIN_TRADES];
  return ["VALIDATION", null, null];
}

// Mirrors src/learning/learning_status.py::compute_learning_status — same
// stage boundaries, same field set, so the dashboard and the HTML report
// never disagree about what stage a mode is in.
export function computeLearningStage(
  evidence: Evidence,
  recommendationsCount: number,
  simulationsCount: number,
  candidatesCount: number,
  promotionEligible: boolean
): LearningStatus {
  const readiness = computeEvidenceReadiness(evidence);
  const [stage, nextStage, nextMin] = stageFor(evidence, readiness.evidenceReadinessPct);

  let tradesToNextStage: number;
  let evidenceGaps: string[];
  if (stage === "BOOTSTRAP") {
    evidenceGaps = bootstrapGaps(evidence, readiness.evidenceReadinessPct);
    tradesToNextStage = Math.max(0, OBSERVATION_MIN_TRADES - evidence.closedTrades);
  } else if (nextMin != null) {
    tradesToNextStage = Math.max(0, nextMin - evidence.closedTrades);
    evidenceGaps = [`${tradesToNextStage} more closed trades`];
  } else {
    tradesToNextStage = 0;
    evidenceGaps = [];
  }
  const reason = nextStage == null ? "Full validation stage reached." : `Need ${evidenceGaps.join(" OR ")} to reach ${nextStage}.`;

  return {
    stage,
    tradesCollected: evidence.closedTrades,
    rejectedTrades: evidence.rejectedOpportunities,
    winningTrades: evidence.winningTrades,
    losingTrades: evidence.losingTrades,
    evidence,
    evidenceReadinessPct: readiness.evidenceReadinessPct,
    dataSufficiencyPct: readiness.components.tradeCoverage,
    recommendationsCount,
    simulationsCount,
    candidatesCount,
    promotionEligible,
    nextStage,
    tradesToNextStage,
    evidenceGaps,
    currentActivity: STAGE_ACTIVITY[stage],
    reason,
  };
}
