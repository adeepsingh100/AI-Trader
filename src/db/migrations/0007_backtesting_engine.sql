-- Event-Driven Backtesting & Walk-Forward Validation Engine. Replays real
-- historical OHLCV chronologically through the same pure pipeline functions
-- live trading uses, simulates realistic order execution, and validates
-- strategy parameters (including adaptive_strategy_versions candidates)
-- against historical trading — see PROJECT_SPEC.md §3c.
--
-- 8 new tables, genuinely more than the last feature collapsed to — unlike
-- that one, most of what's asked for here has no existing equivalent
-- (simulated trades under a run, mark-to-market snapshots, order-lifecycle
-- mechanics, multi-fold rolling results). Where an existing pattern DOES
-- fit, it's reused: metrics are jsonb bundles (matching
-- strategy_simulations.baseline_metrics/candidate_metrics), not wide
-- numeric-column sprawl, and there's no "Simulation Reports" table — a
-- report is generated on demand from the tables below, same precedent as
-- the existing Adaptive Strategy report having no storage table of its own.

-- historical_candles: the raw data cache. No local candle storage existed
-- anywhere before this — every live run always fetched fresh from CoinDCX.
-- `time` is bar-OPEN time (confirmed empirically against the live API), ms
-- epoch, matching CoinDCX's own raw shape exactly (no unit conversion).
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

-- backtest_runs: one row per backtest invocation. symbols is an explicit,
-- user-supplied list — NOT a reconstructed "top-N by turnover" ranking,
-- because CoinDCX has no historical ticker/turnover series to replay and
-- defaulting to today's top symbols over history would be survivorship
-- bias. source_adaptive_strategy_version_id is the natural link for
-- backtesting a pending Adaptive Strategy Engine candidate before a human
-- approves it (nullable — most runs are ad hoc parameter sweeps, not tied
-- to a specific candidate).
create table backtest_runs (
  id                                bigserial primary key,
  name                              text,
  symbols                           text[] not null,
  start_date                        date not null,
  end_date                          date not null,
  warmup_buffer_days                int not null,
  starting_capital                  numeric not null,
  params_json                       jsonb not null,
  use_llm_signal_agent              boolean not null default false,
  source_adaptive_strategy_version_id bigint references adaptive_strategy_versions(id),
  status                            text not null default 'running',  -- 'running' | 'completed' | 'failed'
  created_at                        timestamptz not null default now(),
  completed_at                      timestamptz
);

-- backtest_trades: deliberately SEPARATE from the live `trades` table, not
-- a mode='backtest' row reusing it. trades.mode is woven into live
-- capital_config/daily_pnl/circuit-breaker/promotion-criteria semantics
-- everywhere, and a single historical period gets backtested many times
-- under different params, which `trades` has no run_id concept for.
-- Conflating them risks a bug leaking simulated data into live
-- dashboards/risk state. No FK to strategy_versions — a param-sweep run
-- doesn't correspond to a real live version row, same "don't force a
-- dependency between orthogonal concepts" precedent as
-- adaptive_strategy_versions vs. strategy_versions (see migration 0006).
create table backtest_trades (
  id                      bigserial primary key,
  run_id                  bigint not null references backtest_runs(id),
  symbol                  text not null,
  side                    text not null,
  qty                     numeric not null,
  entry_price             numeric not null,
  exit_price              numeric,
  entry_time              timestamptz not null,
  exit_time               timestamptz,
  holding_duration_seconds int,
  mfe_pct                 numeric,
  mae_pct                 numeric,
  slippage_cost           numeric,
  commission              numeric,
  pnl                     numeric,
  return_pct              numeric,
  risk_reward             numeric,
  exit_reason             text,
  confidence              numeric,
  opportunity_score       numeric,
  market_regime           text,
  created_at              timestamptz not null default now()
);

-- backtest_portfolio_snapshots: mark-to-market equity curve — genuinely
-- new, no existing equivalent (the only existing drawdown,
-- evolution_agent._max_drawdown_pct, walks the trade-pnl sequence, not an
-- intraday equity curve). Persisted at the run's decision-cycle cadence,
-- not every simulation tick, to keep row count sane over a multi-month run
-- (mark-to-market itself still happens every tick, in-memory).
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

-- backtest_execution_history: one row per ORDER-LIFECYCLE EVENT (submitted/
-- filled/partial/rejected/expired/cancelled) — a different grain from
-- backtest_trades (a round-trip position outcome). A limit/stop order can
-- be rejected or expire unfilled and never become a trade at all; a single
-- trade's entry can involve retries. Not a duplicate of backtest_trades.
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

-- backtest_performance_metrics: one row per run, the full Step 6/7 metrics
-- bundle as jsonb — same "bundle, not wide columns" pattern as
-- strategy_simulations.baseline_metrics/candidate_metrics, so a new metric
-- never needs a schema migration.
create table backtest_performance_metrics (
  id         bigserial primary key,
  run_id     bigint not null unique references backtest_runs(id),
  metrics    jsonb not null,
  created_at timestamptz not null default now()
);

-- backtest_walk_forward_folds: real rolling multi-fold validation —
-- deliberately parallel to but SEPARATE from strategy_simulations, which
-- is single-split trade-repartition only (does a weight recommendation
-- separate already-observed outcomes better on a later slice of the SAME
-- trades). This table answers a different question (would this parameter
-- set have made money trading a held-out historical period it never saw)
-- at a genuinely different grain (N rolling folds, not one split).
create table backtest_walk_forward_folds (
  id                  bigserial primary key,
  run_id              bigint not null references backtest_runs(id),
  fold_number         int not null,
  train_window_start  date not null,
  train_window_end    date not null,
  test_window_start   date not null,
  test_window_end     date not null,
  in_sample_metrics   jsonb,
  out_of_sample_metrics jsonb,
  p_value             numeric,
  passed              boolean,
  created_at          timestamptz not null default now()
);

create unique index backtest_walk_forward_folds_run_fold_key
  on backtest_walk_forward_folds (run_id, fold_number);

-- backtest_strategy_comparisons: pairwise run comparison. promotion_recommended
-- is automatic STATUS MARKING only ("reject weak strategies automatically"
-- means auditable classification, never automatic deletion or live
-- application) — matching this session's established human-approval
-- precedent throughout (recommendations, adaptive_strategy_versions).
create table backtest_strategy_comparisons (
  id                     bigserial primary key,
  run_id_a               bigint not null references backtest_runs(id),
  run_id_b               bigint not null references backtest_runs(id),
  metrics_a              jsonb,
  metrics_b              jsonb,
  p_values               jsonb,
  winner                 text,
  promotion_recommended  boolean,
  created_at             timestamptz not null default now()
);

create index backtest_runs_status_idx on backtest_runs (status, created_at);
create index backtest_trades_run_idx on backtest_trades (run_id);
create index backtest_trades_run_symbol_idx on backtest_trades (run_id, symbol);
create index backtest_portfolio_snapshots_run_idx on backtest_portfolio_snapshots (run_id, snapshot_time);
create index backtest_execution_history_run_idx on backtest_execution_history (run_id, event_time);
create index backtest_walk_forward_folds_run_idx on backtest_walk_forward_folds (run_id);
create index backtest_strategy_comparisons_run_a_idx on backtest_strategy_comparisons (run_id_a);
create index backtest_strategy_comparisons_run_b_idx on backtest_strategy_comparisons (run_id_b);

alter table historical_candles enable row level security;
alter table backtest_runs enable row level security;
alter table backtest_trades enable row level security;
alter table backtest_portfolio_snapshots enable row level security;
alter table backtest_execution_history enable row level security;
alter table backtest_performance_metrics enable row level security;
alter table backtest_walk_forward_folds enable row level security;
alter table backtest_strategy_comparisons enable row level security;

create policy "public read" on historical_candles for select using (true);
create policy "public read" on backtest_runs for select using (true);
create policy "public read" on backtest_trades for select using (true);
create policy "public read" on backtest_portfolio_snapshots for select using (true);
create policy "public read" on backtest_execution_history for select using (true);
create policy "public read" on backtest_performance_metrics for select using (true);
create policy "public read" on backtest_walk_forward_folds for select using (true);
create policy "public read" on backtest_strategy_comparisons for select using (true);

grant select on historical_candles to anon, authenticated;
grant select on backtest_runs to anon, authenticated;
grant select on backtest_trades to anon, authenticated;
grant select on backtest_portfolio_snapshots to anon, authenticated;
grant select on backtest_execution_history to anon, authenticated;
grant select on backtest_performance_metrics to anon, authenticated;
grant select on backtest_walk_forward_folds to anon, authenticated;
grant select on backtest_strategy_comparisons to anon, authenticated;
