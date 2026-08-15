export type Mode = "paper" | "real";

export interface CapitalConfig {
  mode: Mode;
  total_capital: number;
  capital_to_use: number;
  daily_profit_target: number;
  max_daily_loss: number;
  position_size_pct: number;
  max_concurrent_positions: number;
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
