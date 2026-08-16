-- Trade Memory + Learning Engine. Every completed trade gets its full
-- entry/exit context stored (extends trades + opportunity_evaluations),
-- plus 5 new tables for bucketed statistics, similarity-search feature
-- importance, confidence-calibration audit, advisory recommendations
-- (never auto-applied), and per-trade self-evaluation. See
-- PROJECT_SPEC.md §3a. Pure statistics, no ML/RL.

alter table trades add column stop_loss_price numeric;
alter table trades add column take_profit_price numeric;
alter table trades add column entry_slippage_pct numeric;
alter table trades add column mfe_pct numeric not null default 0;
alter table trades add column mae_pct numeric not null default 0;
alter table trades add column exit_reason text;     -- 'stop_loss' | 'take_profit' | 'ai_exit' | 'circuit_breaker'
alter table trades add column market_regime text;   -- entry-time classification, see opportunity_scorer.classify_market_regime

create index trades_exit_reason_idx on trades (exit_reason);
create index trades_market_regime_idx on trades (market_regime);

-- one trade can have 2 evaluation rows (entry + an LLM-validated exit); a
-- stop-loss/take-profit sweep exit has no evaluation row at all (no LLM
-- call), so trade_id is the only place this link lives, not the reverse.
alter table opportunity_evaluations add column trade_id bigint references trades(id);
create index opportunity_evaluations_trade_id_idx on opportunity_evaluations (trade_id);

-- learning_statistics: EAV-style, one row per (mode, dimension_type,
-- dimension_value) bucket, upserted in place on recompute. dimension_type
-- in ('symbol','market_regime','opportunity_score_bucket',
-- 'confidence_bucket','strategy_version','weekday','hour'). No
-- recovery_factor column: at this codebase's simplification level it's
-- numerically identical to calmar_ratio, storing both would duplicate one
-- number under two names.
create table learning_statistics (
  id                      bigserial primary key,
  mode                    text not null,
  dimension_type          text not null,
  dimension_value         text not null,
  trades_count            int not null default 0,
  win_rate                numeric,
  avg_profit              numeric,
  avg_loss                numeric,
  profit_factor           numeric,
  expectancy              numeric,
  avg_holding_time_seconds numeric,
  max_drawdown_pct        numeric,
  sharpe_ratio            numeric,
  sortino_ratio           numeric,
  calmar_ratio            numeric,
  computed_at             timestamptz not null default now(),
  unique (mode, dimension_type, dimension_value)
);

-- feature_importance: point-biserial correlation between each Feature
-- Engine key (primary timeframe only) and trade win/loss outcome.
create table feature_importance (
  id               bigserial primary key,
  mode             text not null,
  feature_name     text not null,
  correlation_score numeric,
  sample_count     int not null default 0,
  computed_at      timestamptz not null default now(),
  unique (mode, feature_name)
);

-- confidence_calibration: an audit log (one row per entry-validation
-- call), not aggregate stats — what calibration was actually applied to
-- a specific decision, distinct purpose from learning_statistics.
create table confidence_calibration (
  id                        bigserial primary key,
  opportunity_evaluation_id bigint not null references opportunity_evaluations(id),
  ai_confidence             numeric,
  historical_confidence     numeric,
  ai_weight                 numeric,
  historical_weight         numeric,
  final_confidence          numeric,
  similar_trades_count      int not null default 0,
  created_at                timestamptz not null default now()
);

-- recommendations: advisory only, human-approval required, never
-- auto-applied to config. Append-only (not upserted) so a human's
-- 'dismissed' status on an old row survives — idempotency is enforced in
-- application code (skip inserting if the latest row's recommended_value
-- hasn't moved materially), not by a DB constraint.
create table recommendations (
  id                bigserial primary key,
  mode              text not null,
  metric_name       text not null,
  current_value     numeric,
  recommended_value numeric,
  rationale         text,
  sample_size       int not null default 0,
  status            text not null default 'pending',  -- 'pending' | 'reviewed' | 'dismissed'
  created_at        timestamptz not null default now()
);

-- trade_evaluations: 1:1 self-evaluation child of trades — was the
-- predicted confidence/score accurate, checked after the fact.
create table trade_evaluations (
  trade_id                     bigint primary key references trades(id),
  predicted_confidence         numeric,
  predicted_opportunity_score  numeric,
  actual_outcome_won           boolean not null,
  confidence_was_accurate      boolean,
  opportunity_score_was_accurate boolean,
  risk_assessment              text,
  stop_loss_assessment         text,
  target_assessment            text,
  evaluated_at                 timestamptz not null default now()
);

create index learning_statistics_lookup_idx on learning_statistics (mode, dimension_type);
create index feature_importance_mode_idx on feature_importance (mode);
create index confidence_calibration_eval_idx on confidence_calibration (opportunity_evaluation_id);
create index recommendations_mode_metric_idx on recommendations (mode, metric_name);

alter table learning_statistics enable row level security;
alter table feature_importance enable row level security;
alter table confidence_calibration enable row level security;
alter table recommendations enable row level security;
alter table trade_evaluations enable row level security;

create policy "public read" on learning_statistics for select using (true);
create policy "public read" on feature_importance for select using (true);
create policy "public read" on confidence_calibration for select using (true);
create policy "public read" on recommendations for select using (true);
create policy "public read" on trade_evaluations for select using (true);

grant select on learning_statistics to anon, authenticated;
grant select on feature_importance to anon, authenticated;
grant select on confidence_calibration to anon, authenticated;
grant select on recommendations to anon, authenticated;
grant select on trade_evaluations to anon, authenticated;
