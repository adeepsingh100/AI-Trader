-- Adaptive Strategy Intelligence Engine. Closes the loop from the Learning
-- Engine's statistics into advisory recommendations, walk-forward
-- validation, and versioned candidate parameter sets — all human-approved,
-- nothing auto-applied to live trading. See PROJECT_SPEC.md §3b.
--
-- 3 new/extended tables, not the literal 8 the feature was specced with —
-- "Feature Weight History"/"Threshold History" are just
-- recommendations WHERE category IN ('weight','threshold') (already
-- append-only), "Market Regime Statistics" is already
-- learning_statistics WHERE dimension_type='market_regime', and a would-be
-- "Adaptive Strategies" table for "the currently active parameter set" is
-- deliberately not built (see adaptive_strategy_versions comment below) —
-- same "don't store one fact under two names" precedent already applied to
-- calmar_ratio/recovery_factor and learning_statistics/market_regimes.

-- feature_importance gains a timeframe dimension (Step 6: per-timeframe
-- specialization, and also the home for cached sub-score correlation
-- weights consumed live by trade_memory.find_similar_trades — see
-- feature_name convention below). It's a fully derived/recomputable
-- nightly cache, so old rows are truncated rather than backfilled a
-- timeframe value.
truncate table feature_importance;
alter table feature_importance add column timeframe text not null default 'unknown';
alter table feature_importance alter column timeframe drop default;
alter table feature_importance drop constraint feature_importance_mode_feature_name_key;
alter table feature_importance add constraint feature_importance_mode_feature_name_timeframe_key
  unique (mode, feature_name, timeframe);
-- feature_name convention: raw Feature Engine keys (rsi, adx, ...) use their
-- own timeframe ('1m'/'15m'/'1h'/'1d'); the 5 opportunity-scorer sub-score
-- names (trend_score, momentum_score, volume_score, volatility_score,
-- risk_score) always use the explicit sentinel 'blended' (a real, meaningful
-- value — correlation computed across the already-timeframe-blended score —
-- not a null special case).

-- confidence_calibration gains the adaptive modifier chain (Step 7) so
-- every stage — not just the final blended number — is individually
-- auditable, matching the objective's "measurable, explainable,
-- auditable" requirement. This chain runs automatically every cycle
-- (unlike the rest of this migration's advisory tables) — it's an
-- extension of the ALREADY-automatic calibrate_confidence gate, inert in
-- practice while MIN_FINAL_CONFIDENCE stays at its default 0.
alter table confidence_calibration add column regime_modifier numeric;
alter table confidence_calibration add column symbol_modifier numeric;
alter table confidence_calibration add column recent_performance_modifier numeric;

-- recommendations gains category/confidence/evidence/batch_id so the
-- existing advisory table (never auto-applied, human-approved in Supabase)
-- can carry weight/regime/symbol recommendations alongside the existing
-- threshold ones, with real statistical backing and grouping.
alter table recommendations add column category text not null default 'threshold';
alter table recommendations alter column category drop default;
alter table recommendations add column confidence numeric;  -- (1 - p_value) * 100 from the z-test below
alter table recommendations add column evidence jsonb;       -- supporting trade ids / affected bucket refs, variable per category
alter table recommendations add column batch_id uuid;        -- groups co-generated rows (e.g. the 5 weight recommendations from one call)

create index recommendations_mode_category_status_idx on recommendations (mode, category, status);

-- strategy_simulations: walk-forward train/test result for one
-- recommendation batch — this table IS "Walk Forward Results" too (same
-- computation reported together, not a second artifact).
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
  created_at              timestamptz not null default now()
);

-- adaptive_strategy_versions: versions QUANTITATIVE PARAMETERS (a params_json
-- snapshot of the tunable adaptive constants) — orthogonal to the existing
-- strategy_versions table, which versions LLM PROMPT TEXT. Created lazily,
-- only for simulations that pass (strategy_simulations.passed = true), so
-- this table only ever holds genuine candidates, never a row per nightly
-- batch nobody looked at. Never overwritten; rollback is a new row's status
-- change, not a mutation of an earlier one. No separate "currently active"
-- table — `WHERE status='approved' ORDER BY created_at DESC LIMIT 1`
-- answers that honestly, since auto-deploy is out of scope this phase and a
-- denormalized "active" row would drift from the real source of truth
-- (env vars a human still updates by hand).
create table adaptive_strategy_versions (
  id                          bigserial primary key,
  mode                        text not null,
  version_number              int not null,
  params_json                 jsonb not null,
  source_recommendation_batch_id uuid,
  source_simulation_id        bigint references strategy_simulations(id),
  status                      text not null default 'candidate',  -- 'candidate' | 'approved' | 'rolled_back'
  notes                       text,
  created_at                  timestamptz not null default now()
);

create index strategy_simulations_mode_idx on strategy_simulations (mode, created_at);
create index strategy_simulations_batch_idx on strategy_simulations (recommendation_batch_id);
create index adaptive_strategy_versions_mode_status_idx on adaptive_strategy_versions (mode, status);

alter table strategy_simulations enable row level security;
alter table adaptive_strategy_versions enable row level security;

create policy "public read" on strategy_simulations for select using (true);
create policy "public read" on adaptive_strategy_versions for select using (true);

grant select on strategy_simulations to anon, authenticated;
grant select on adaptive_strategy_versions to anon, authenticated;

