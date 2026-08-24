-- Neon init — hand-authored final-state schema, consolidating
-- 0001_init.sql through 0011_promotion_audit.sql for a fresh database.
-- Not a mechanical concatenation: several of those files did
-- transitional churn (e.g. 0006's truncate+multiple alters on
-- feature_importance) that's meaningless on a table that never had the
-- old state — this file is what falls out of applying all 11 in order
-- and reading off the final column set. 0009's redundant index is
-- simply never created here (equivalent end state to create-then-drop).
--
-- Every RLS/grant/auth.role() statement from the original 11 files is
-- deliberately dropped — meaningless without Supabase's PostgREST/Auth
-- layer (no anon/authenticated roles, no auth schema exist on Neon).
-- Public-read access now lives in the dashboard's API routes instead;
-- the one write path (capital_config, via /api/config) is gated by the
-- dashboard's own password check, not a DB policy.
--
-- 0001-0011 are untouched history, applied to the old Supabase project
-- only — do not edit them. Any future schema change after this file
-- becomes 0013_....sql in this same directory.

create table capital_config (
  mode                    text primary key,       -- 'paper' | 'real'
  total_capital           numeric not null,
  capital_to_use          numeric not null,
  daily_profit_target     numeric not null,
  max_daily_loss          numeric not null,
  position_size_pct       numeric not null default 10,
  max_concurrent_positions int not null default 5,
  paused                  boolean not null default false,
  sizing_mode             text not null default 'flat',
  updated_at              timestamptz not null default now()
);

create table strategy_versions (
  id                 bigserial primary key,
  version_number     int not null,
  prompt_text        text not null,
  params_json        jsonb not null default '{}',
  promoted_to_real   boolean not null default false,
  notes              text,
  status             text not null default 'active',
  promotion_eligible boolean not null default false,
  created_at         timestamptz not null default now()
);

create table trades (
  id                 bigserial primary key,
  mode               text not null,
  version_id         bigint not null references strategy_versions(id),
  symbol             text not null,
  side               text not null,
  qty                numeric not null,
  entry_price        numeric not null,
  exit_price         numeric,
  pnl                numeric,
  fees               numeric not null default 0,
  status             text not null,
  opened_at          timestamptz not null default now(),
  closed_at          timestamptz,
  reasoning_text     text,
  stop_loss_price    numeric,
  take_profit_price  numeric,
  entry_slippage_pct numeric,
  mfe_pct            numeric not null default 0,
  mae_pct            numeric not null default 0,
  exit_reason        text,     -- 'stop_loss' | 'take_profit' | 'ai_exit' | 'circuit_breaker'
  market_regime      text      -- entry-time classification
);

create table daily_pnl (
  date                      date not null,
  mode                      text not null,
  realized_pnl              numeric not null default 0,
  trades_count              int not null default 0,
  target_hit                boolean not null default false,
  circuit_breaker_triggered boolean not null default false,
  primary key (date, mode)
);

create table agent_logs (
  id               bigserial primary key,
  timestamp        timestamptz not null default now(),
  agent_name       text not null,
  level            text not null,
  message          text not null,
  raw_llm_response jsonb
);

create table model_usage (
  id              bigserial primary key,
  timestamp       timestamptz not null default now(),
  model_used      text not null,
  fallback_reason text,
  latency_ms      int not null,
  success         boolean not null
);

create table opportunity_evaluations (
  id                  bigserial primary key,
  timestamp           timestamptz not null default now(),
  mode                text not null,
  symbol              text not null,
  version_id          bigint not null references strategy_versions(id),
  features            jsonb not null,
  trend_score         numeric,
  momentum_score      numeric,
  volume_score        numeric,
  volatility_score    numeric,
  risk_score          numeric,
  opportunity_score   numeric,
  llm_decision        text,
  llm_reasoning       text,
  llm_raw_response    jsonb,
  risk_manager_result text,
  final_decision      text not null,
  reason              text,
  trade_id            bigint references trades(id),
  config_version      text,
  market_regime       text
);

create table learning_statistics (
  id                       bigserial primary key,
  mode                     text not null,
  dimension_type           text not null,
  dimension_value          text not null,
  trades_count             int not null default 0,
  win_rate                 numeric,
  avg_profit               numeric,
  avg_loss                 numeric,
  profit_factor            numeric,
  expectancy               numeric,
  avg_holding_time_seconds numeric,
  max_drawdown_pct         numeric,
  sharpe_ratio             numeric,
  sortino_ratio            numeric,
  calmar_ratio             numeric,
  computed_at              timestamptz not null default now(),
  unique (mode, dimension_type, dimension_value)
);

create table feature_importance (
  id                bigserial primary key,
  mode              text not null,
  feature_name      text not null,
  timeframe         text not null,
  correlation_score numeric,
  sample_count      int not null default 0,
  computed_at       timestamptz not null default now(),
  unique (mode, feature_name, timeframe)
);

create table confidence_calibration (
  id                          bigserial primary key,
  opportunity_evaluation_id   bigint not null references opportunity_evaluations(id),
  ai_confidence               numeric,
  historical_confidence       numeric,
  ai_weight                   numeric,
  historical_weight           numeric,
  final_confidence            numeric,
  similar_trades_count        int not null default 0,
  regime_modifier             numeric,
  symbol_modifier             numeric,
  recent_performance_modifier numeric,
  created_at                  timestamptz not null default now()
);

create table recommendations (
  id                bigserial primary key,
  mode              text not null,
  metric_name       text not null,
  current_value     numeric,
  recommended_value numeric,
  rationale         text,
  sample_size       int not null default 0,
  status            text not null default 'pending',  -- 'pending' | 'reviewed' | 'dismissed'
  category          text not null default 'threshold',
  confidence        numeric,
  evidence          jsonb,
  batch_id          uuid,
  created_at        timestamptz not null default now()
);

create table trade_evaluations (
  trade_id                        bigint primary key references trades(id),
  predicted_confidence            numeric,
  predicted_opportunity_score     numeric,
  actual_outcome_won              boolean not null,
  confidence_was_accurate         boolean,
  opportunity_score_was_accurate  boolean,
  risk_assessment                 text,
  stop_loss_assessment            text,
  target_assessment               text,
  evaluated_at                    timestamptz not null default now()
);

create table strategy_simulations (
  id                      bigserial primary key,
  recommendation_batch_id uuid,
  mode                    text not null,
  train_window_start      timestamptz not null,
  train_window_end        timestamptz not null,
  test_window_start       timestamptz not null,
  test_window_end         timestamptz not null,
  baseline_metrics        jsonb,
  candidate_metrics       jsonb,
  p_value                 numeric,
  passed                  boolean not null,
  research_note           text,
  validation_detail       jsonb,
  created_at              timestamptz not null default now()
);

create table adaptive_strategy_versions (
  id                              bigserial primary key,
  mode                            text not null,
  version_number                  int not null,
  params_json                     jsonb not null,
  source_recommendation_batch_id  uuid,
  source_simulation_id            bigint references strategy_simulations(id),
  status                          text not null default 'candidate',  -- 'candidate' | 'approved' | 'rolled_back'
  notes                           text,
  fitness_score                   numeric,
  created_at                      timestamptz not null default now()
);

create table historical_candles (
  id         bigserial primary key,
  pair       text not null,
  interval   text not null,
  time       bigint not null,
  open       numeric not null,
  high       numeric not null,
  low        numeric not null,
  close      numeric not null,
  volume     numeric not null,
  created_at timestamptz not null default now()
);

create unique index historical_candles_pair_interval_time_key
  on historical_candles (pair, interval, time);

create table backtest_runs (
  id                                   bigserial primary key,
  name                                 text,
  symbols                              text[] not null,
  start_date                           date not null,
  end_date                             date not null,
  warmup_buffer_days                   int not null,
  starting_capital                     numeric not null,
  params_json                          jsonb not null,
  use_llm_signal_agent                 boolean not null default false,
  source_adaptive_strategy_version_id  bigint references adaptive_strategy_versions(id),
  status                               text not null default 'running',  -- 'running' | 'completed' | 'failed'
  created_at                           timestamptz not null default now(),
  completed_at                         timestamptz
);

create table backtest_trades (
  id                       bigserial primary key,
  run_id                   bigint not null references backtest_runs(id),
  symbol                   text not null,
  side                     text not null,
  qty                      numeric not null,
  entry_price              numeric not null,
  exit_price               numeric,
  entry_time               timestamptz not null,
  exit_time                timestamptz,
  holding_duration_seconds int,
  mfe_pct                  numeric,
  mae_pct                  numeric,
  slippage_cost            numeric,
  commission               numeric,
  pnl                      numeric,
  return_pct               numeric,
  risk_reward              numeric,
  exit_reason              text,
  confidence               numeric,
  opportunity_score        numeric,
  market_regime            text,
  created_at               timestamptz not null default now()
);

create table backtest_portfolio_snapshots (
  id                   bigserial primary key,
  run_id               bigint not null references backtest_runs(id),
  snapshot_time        timestamptz not null,
  cash                 numeric not null,
  equity               numeric not null,
  unrealized_pnl       numeric,
  realized_pnl         numeric,
  exposure_pct         numeric,
  open_positions_count int,
  created_at           timestamptz not null default now()
);

create unique index backtest_portfolio_snapshots_run_time_key
  on backtest_portfolio_snapshots (run_id, snapshot_time);

create table backtest_execution_history (
  id               bigserial primary key,
  run_id           bigint not null references backtest_runs(id),
  symbol           text not null,
  order_type       text not null,  -- 'market' | 'limit' | 'stop' | 'stop_limit' | 'trailing_stop'
  side             text not null,
  requested_qty    numeric not null,
  requested_price  numeric,
  status           text not null,  -- 'submitted' | 'filled' | 'partial' | 'rejected' | 'expired' | 'cancelled'
  filled_qty       numeric,
  filled_price     numeric,
  rejection_reason text,
  event_time       timestamptz not null,
  created_at       timestamptz not null default now()
);

create table backtest_performance_metrics (
  id         bigserial primary key,
  run_id     bigint not null unique references backtest_runs(id),
  metrics    jsonb not null,
  created_at timestamptz not null default now()
);

create table backtest_walk_forward_folds (
  id                    bigserial primary key,
  run_id                bigint not null references backtest_runs(id),
  fold_number           int not null,
  train_window_start    date not null,
  train_window_end      date not null,
  test_window_start     date not null,
  test_window_end       date not null,
  in_sample_metrics     jsonb,
  out_of_sample_metrics jsonb,
  p_value               numeric,
  passed                boolean,
  created_at            timestamptz not null default now()
);

create unique index backtest_walk_forward_folds_run_fold_key
  on backtest_walk_forward_folds (run_id, fold_number);

create table backtest_strategy_comparisons (
  id                    bigserial primary key,
  run_id_a              bigint not null references backtest_runs(id),
  run_id_b              bigint not null references backtest_runs(id),
  metrics_a             jsonb,
  metrics_b             jsonb,
  p_values              jsonb,
  winner                text,
  promotion_recommended boolean,
  created_at            timestamptz not null default now()
);

create table data_quality_log (
  id         bigserial primary key,
  pair       text not null,
  interval   text not null,
  source     text not null,  -- 'live' | 'backtest'
  issue_type text not null,
  severity   text not null,  -- 'ignore' | 'warn' | 'reject' | 'quarantine'
  detail     jsonb not null default '{}',
  repaired   boolean not null default false,
  created_at timestamptz not null default now()
);

create table drift_alerts (
  id             bigserial primary key,
  component      text not null,
  drift_type     text not null,
  severity       text not null,  -- 'warning' | 'critical'
  baseline_value numeric,
  recent_value   numeric,
  detail         jsonb not null default '{}',
  detected_at    timestamptz not null default now()
);

create table strategy_health_scores (
  id                  bigserial primary key,
  strategy_version_id bigint not null references strategy_versions(id),
  health_score        numeric,
  tier                text not null,  -- 'excellent' | 'good' | 'warning' | 'critical' | 'unknown'
  breakdown           jsonb not null default '{}',
  computed_at         timestamptz not null default now()
);

create table system_metrics (
  id          bigserial primary key,
  component   text not null,
  metric_name text not null,
  value       numeric,
  metadata    jsonb not null default '{}',
  recorded_at timestamptz not null default now()
);

create table circuit_breaker_state (
  component            text primary key,  -- 'coindcx_api' | 'db' | 'llm'
  consecutive_failures int not null default 0,
  tripped_until        bigint,  -- epoch ms, null = not tripped
  updated_at           timestamptz not null default now()
);

create table promotion_audit (
  id                   bigserial primary key,
  mode                 text not null,
  event_type           text not null,  -- 'evaluation' | 'promotion' | 'rollback'
  candidate_version_id bigint references strategy_versions(id),
  previous_champion_id bigint references strategy_versions(id),
  new_champion_id      bigint references strategy_versions(id),
  decision             text not null,  -- 'PROMOTE' | 'REJECT' | 'EXTEND_VALIDATION'
  promotion_score      numeric,
  gates                jsonb not null default '{}',
  breakdown            jsonb not null default '{}',
  reasons              jsonb not null default '[]',
  created_at           timestamptz not null default now()
);

-- Indexes (from all 11 originals; the one redundant index from 0007,
-- dropped again in 0009, is simply never created here)
create index trades_mode_status_idx on trades (mode, status);
create index trades_opened_at_idx on trades (opened_at);
create index trades_exit_reason_idx on trades (exit_reason);
create index trades_market_regime_idx on trades (market_regime);
create index agent_logs_timestamp_idx on agent_logs (timestamp);
create index model_usage_timestamp_idx on model_usage (timestamp);
create index opportunity_evaluations_mode_timestamp_idx on opportunity_evaluations (mode, timestamp);
create index opportunity_evaluations_symbol_idx on opportunity_evaluations (symbol);
create index opportunity_evaluations_trade_id_idx on opportunity_evaluations (trade_id);
create index learning_statistics_lookup_idx on learning_statistics (mode, dimension_type);
create index feature_importance_mode_idx on feature_importance (mode);
create index confidence_calibration_eval_idx on confidence_calibration (opportunity_evaluation_id);
create index recommendations_mode_category_status_idx on recommendations (mode, category, status);
create index strategy_simulations_mode_idx on strategy_simulations (mode, created_at);
create index strategy_simulations_batch_idx on strategy_simulations (recommendation_batch_id);
create index adaptive_strategy_versions_mode_status_idx on adaptive_strategy_versions (mode, status);
create index backtest_runs_status_idx on backtest_runs (status, created_at);
create index backtest_trades_run_idx on backtest_trades (run_id);
create index backtest_trades_run_symbol_idx on backtest_trades (run_id, symbol);
create index backtest_execution_history_run_idx on backtest_execution_history (run_id, event_time);
create index backtest_walk_forward_folds_run_idx on backtest_walk_forward_folds (run_id);
create index backtest_strategy_comparisons_run_a_idx on backtest_strategy_comparisons (run_id_a);
create index backtest_strategy_comparisons_run_b_idx on backtest_strategy_comparisons (run_id_b);
create index data_quality_log_pair_interval_idx on data_quality_log (pair, interval, created_at);
create index data_quality_log_source_idx on data_quality_log (source, created_at);
create index drift_alerts_component_idx on drift_alerts (component, detected_at);
create index strategy_health_scores_version_idx on strategy_health_scores (strategy_version_id, computed_at);
create index system_metrics_component_idx on system_metrics (component, recorded_at);
create index system_metrics_name_idx on system_metrics (metric_name, recorded_at);
create index promotion_audit_mode_created_idx on promotion_audit (mode, created_at);
create index promotion_audit_event_type_idx on promotion_audit (event_type, created_at);
