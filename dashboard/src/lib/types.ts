export type Mode = "paper" | "real";

export interface CapitalConfig {
  mode: Mode;
  total_capital: number;
  capital_to_use: number;
  daily_profit_target: number;
  max_daily_loss: number;
  position_size_pct: number;
  max_concurrent_positions: number;
  paused: boolean;
  updated_at: string;
}

export interface StrategyVersion {
  id: number;
  version_number: number;
  prompt_text: string;
  params_json: Record<string, unknown>;
  promoted_to_real: boolean;
  notes: string | null;
  created_at: string;
}

export interface Trade {
  id: number;
  mode: Mode;
  version_id: number;
  symbol: string;
  side: "buy" | "sell";
  qty: number;
  entry_price: number;
  exit_price: number | null;
  pnl: number | null;
  fees: number;
  status: "open" | "closed" | "flattened";
  opened_at: string;
  closed_at: string | null;
  reasoning_text: string | null;
}

export interface DailyPnl {
  date: string;
  mode: Mode;
  realized_pnl: number;
  trades_count: number;
  target_hit: boolean;
  circuit_breaker_triggered: boolean;
}

export interface ModelUsage {
  id: number;
  timestamp: string;
  model_used: string;
  fallback_reason: string | null;
  latency_ms: number;
  success: boolean;
}

export interface LearningStatistic {
  dimension_type: string;
  dimension_value: string;
  expectancy: number | null;
  trades_count: number;
}

export interface OpportunityEvaluationRow {
  symbol: string | null;
  market_regime: string | null;
  timestamp: string | null;
  final_decision: string | null;
  llm_decision: string | null;
  reason: string | null;
  risk_manager_result: string | null;
}

export interface Recommendation {
  category: string;
  metric_name: string;
  current_value: number | null;
  recommended_value: number | null;
  confidence: number | null;
  sample_size: number;
  rationale: string | null;
  status: string;
  created_at: string;
}

export interface StrategySimulation {
  id: number;
  created_at: string;
  passed: boolean;
  p_value: number | null;
  research_note: string | null;
}

export interface AdaptiveStrategyVersion {
  version_number: number;
  status: string;
  fitness_score: number | null;
  notes: string | null;
  created_at: string;
  source_simulation_id: number | null;
}

export type LearningStage = "BOOTSTRAP" | "OBSERVATION" | "HYPOTHESIS" | "SIMULATION" | "VALIDATION";

export interface Evidence {
  closedTrades: number;
  winningTrades: number;
  losingTrades: number;
  rejectedOpportunities: number;
  candidateOpportunities: number;
  symbolsCovered: number;
  marketRegimesCovered: number;
  tradingHoursCovered: number;
  featureCoveragePct: number;
  confidenceCoveragePct: number;
  learningCoveragePct: number;
  symbolsRarelyQualifying: { symbol: string; seen: number; rejectRatePct: number }[];
  regimesWithNoCandidates: string[];
}

export interface LearningStatus {
  stage: LearningStage;
  tradesCollected: number;
  rejectedTrades: number;
  winningTrades: number;
  losingTrades: number;
  evidence: Evidence;
  evidenceReadinessPct: number;
  dataSufficiencyPct: number;
  recommendationsCount: number;
  simulationsCount: number;
  candidatesCount: number;
  promotionEligible: boolean;
  nextStage: LearningStage | null;
  tradesToNextStage: number;
  evidenceGaps: string[];
  currentActivity: string;
  reason: string;
}
