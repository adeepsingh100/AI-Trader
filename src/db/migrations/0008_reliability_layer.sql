-- Institutional Reliability Layer — see PROJECT_SPEC.md §3d. Purely
-- additive: 5 new tables + 4 new columns on existing tables, all with
-- safe defaults that preserve today's exact behavior until a human
-- deliberately opts in (sizing_mode='flat', strategy_versions.status=
-- 'active' by default — nothing here changes live trading on its own).
--
-- IMPORTANT deployment-order note, unique among this repo's migrations so
-- far: get_latest_version()/get_latest_promoted_version() in
-- src/db/models.py now filter on strategy_versions.status. Unlike prior
-- migrations (which only left a NEW feature dark until run), THIS ONE
-- MUST be applied before the corresponding code deploys — those two
-- functions gate every live trading cycle, and querying a column that
-- doesn't exist yet would error on every run, not just leave a feature
-- unavailable.

-- data_quality_log: Market Data Quality Engine + Data Repair Engine
-- (src/data_quality/). One row per ISSUE found, not per candle — the
-- audit trail for every reject/repair, live and backtest ingestion alike.
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

create index data_quality_log_pair_interval_idx on data_quality_log (pair, interval, created_at);
create index data_quality_log_source_idx on data_quality_log (source, created_at);

-- drift_alerts: Feature Drift Detection (src/learning/drift_detection.py).
-- Advisory only, same as every other src/learning/ output.
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

create index drift_alerts_component_idx on drift_alerts (component, detected_at);

-- strategy_health_scores: Strategy Health Engine
-- (src/learning/strategy_health.py). One row per computation run per
-- version (a history, not just a latest-value table) — computed_at lets a
-- reader see the trend, not just the current tier.
create table strategy_health_scores (
  id                  bigserial primary key,
  strategy_version_id bigint not null references strategy_versions(id),
  health_score        numeric,
  tier                text not null,  -- 'excellent' | 'good' | 'warning' | 'critical' | 'unknown'
  breakdown           jsonb not null default '{}',
  computed_at         timestamptz not null default now()
);

create index strategy_health_scores_version_idx on strategy_health_scores (strategy_version_id, computed_at);

-- system_metrics: Production Monitoring + Self-Diagnostics
-- (src/monitoring/). One generic table, not N single-purpose ones —
-- matching the jsonb-bundle precedent elsewhere in this schema.
create table system_metrics (
  id           bigserial primary key,
  component    text not null,
  metric_name  text not null,
  value        numeric,
  metadata     jsonb not null default '{}',
  recorded_at  timestamptz not null default now()
);

create index system_metrics_component_idx on system_metrics (component, recorded_at);
create index system_metrics_name_idx on system_metrics (metric_name, recorded_at);

-- circuit_breaker_state: src/resilience.py. DB-backed so a trip survives
-- across cron invocations (each GitHub Actions run is a fresh process).
create table circuit_breaker_state (
  component            text primary key,  -- 'coindcx_api' | 'supabase' | 'llm'
  consecutive_failures int not null default 0,
  tripped_until        bigint,  -- epoch ms, null = not tripped
  updated_at           timestamptz not null default now()
);

-- capital_config.sizing_mode: Capital Allocation Engine
-- (src/portfolio/capital_allocation.py). Default 'flat' = today's exact
-- capital_to_use*position_size_pct/100 formula, byte-identical — a human
-- flips a mode's row to 'dynamic' in Supabase after reviewing behavior in
-- paper first, nothing in code does this automatically.
alter table capital_config add column sizing_mode text not null default 'flat';

-- strategy_versions.status: Strategy Health Engine auto-suspension.
-- Status-only marking, never a delete — a human can always flip a
-- suspended version back to active in Supabase.
alter table strategy_versions add column status text not null default 'active';

-- opportunity_evaluations: the two fields Audit System (Step 9) needed
-- that weren't already columns — everything else it asks for
-- (timestamp/component/input/decision/output/reason/strategy-version/
-- confidence/trade-id) was already captured here or on
-- confidence_calibration/trades, so src/audit/trail.py reads those three
-- tables rather than adding a new write path.
alter table opportunity_evaluations add column config_version text;
alter table opportunity_evaluations add column market_regime text;

alter table data_quality_log enable row level security;
alter table drift_alerts enable row level security;
alter table strategy_health_scores enable row level security;
alter table system_metrics enable row level security;
alter table circuit_breaker_state enable row level security;

create policy "public read" on data_quality_log for select using (true);
create policy "public read" on drift_alerts for select using (true);
create policy "public read" on strategy_health_scores for select using (true);
create policy "public read" on system_metrics for select using (true);
create policy "public read" on circuit_breaker_state for select using (true);

grant select on data_quality_log to anon, authenticated;
grant select on drift_alerts to anon, authenticated;
grant select on strategy_health_scores to anon, authenticated;
grant select on system_metrics to anon, authenticated;
grant select on circuit_breaker_state to anon, authenticated;
