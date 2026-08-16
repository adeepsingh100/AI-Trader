# PROJECT_SPEC.md

Multi-agent AI crypto trading bot on CoinDCX (INR pairs). Two modes —
Paper Trading and Real Trading — share one learning/strategy engine.
Real trading only ever runs a strategy version that has been promoted
out of paper trading.

Decisions locked in during scoping (see "Open decisions resolved" below)
are binding unless the user says otherwise. Everything marked
**configurable** lives in env vars or DB config rows, not hardcoded.

---

## 1. Goal

- **Paper Trading**: simulated fills, no real money, proving ground for
  strategy versions.
- **Real Trading**: live signed orders on CoinDCX, gated to only run a
  `promoted_to_real` strategy version.
- Both modes run the same agent pipeline and log to the same schema, so
  paper and real performance are directly comparable.

## 2. Capital & risk rules

- Each mode has its own capital config (`capital_config` table, one row
  per mode). Paper capital and real capital are independent numbers.
- For real trading, `capital_to_use` may be less than the wallet's actual
  balance (e.g. wallet ₹10,000, `capital_to_use` ₹5,000) — Risk Manager
  never sizes positions against more than `capital_to_use`.
- Daily profit target is a soft goal — the bot aims for it, accepts less
  or zero on a bad day. No forced trading to "catch up" to target.
- Daily max-loss is a **hard circuit breaker**, not a guarantee: once
  realized loss for the day hits `max_daily_loss`, the bot stops opening
  new positions **and flattens all open positions immediately**
  (market-close every open trade for that mode, logged to `trades` and
  `daily_pnl.circuit_breaker_triggered = true`). This bounds *realized*
  loss for the day to (approximately) the threshold plus the slippage on
  the flattening orders — it cannot guarantee zero loss, since markets
  gap and orders can fail. Once triggered, no new positions open again
  until the next trading day (00:00 IST rollover).
- Real trading only runs a strategy version where
  `strategy_versions.promoted_to_real = true`. Promotion is decided by
  the Evolution Agent against **configurable** criteria (env vars, not a
  DB table — these change rarely and don't need versioning):
  - `PROMOTION_MIN_PAPER_DAYS` (default 14) — minimum days of paper
    trading history for the version.
  - `PROMOTION_MIN_CUMULATIVE_PNL` (default 0) — cumulative paper PnL
    over that window must exceed this.
  - `PROMOTION_MAX_DRAWDOWN_PCT` (default 15) — max peak-to-trough
    drawdown over the window must stay under this.

## 3. Multi-agent architecture

All agents are separate, independently testable Python modules under
`src/agents/` (plus `src/features/` for the deterministic scoring
pipeline). One full cycle runs to completion and exits — no long-running
process (see §7):

```
Data Agent → Feature Engine → Opportunity Scorer → Candidate Filter
    → Signal Agent (LLM validation) → Risk Manager → Execution Agent → log
```

**Quant-first, not LLM-first**: the LLM never picks direction and never
sees raw candles. A deterministic scorer (zero AI, zero randomness) ranks
every scanned symbol; only the top candidates and deteriorating held
positions reach the LLM at all, as a curated validate/reject gate. This
is what actually bounds LLM call volume per cycle — not the scan
breadth (`n_symbols`, still top-10-by-turnover), which stays cheap
because scoring is pure math.

### Data Agent (`src/agents/data_agent.py`)
- Pulls CoinDCX's public market endpoints (ticker, candles).
- Each cycle: fetches 24h ticker volume for all INR pairs, selects the
  **top 10 by volume** (Resolved decision — see §9: dynamic
  top-10-by-volume, not a fixed pair list), pulls candles for each of
  `FEATURE_TIMEFRAMES` (**configurable**, default `5m,15m,1h,4h`) at
  `FEATURE_CANDLE_LIMIT` (default 250) candles per timeframe.
- No orderbook fetch — nothing in the pipeline consumes book depth
  (`PaperExecutionAgent` uses a flat slippage bps, `RealExecutionAgent`
  trades at market), so it was dropped as a dead API call.
- No auth required (public endpoints only).

### Feature Engine (`src/features/feature_engine.py`)
- Pure functions, no trading decisions: EMA (20/50/100/200), RSI, MACD +
  histogram, Stochastic RSI, ATR (+ ATR%), Bollinger Band width (as a
  %, comparable across differently-priced symbols), relative volume,
  volume spike, OBV (+ short-window rising/falling), support/resistance
  (lookback min/max), ADX/+DI/-DI, and a volatility regime bucket
  (low/medium/high). Every period/threshold is a named `src/config.py`
  constant (`RSI_PERIOD`, `ATR_PERIOD`, `EMA_TREND_PERIOD_1..4`, etc.) —
  nothing is a bare literal.
- Computed once per configured timeframe (`compute_multi_timeframe_features`)
  from that timeframe's own candles alone. Never raises on short
  history — every indicator degrades to `None` instead, so a thin/new
  listing doesn't crash the cycle.

### Opportunity Scorer (`src/features/opportunity_scorer.py`)
- Deterministic, zero AI/LLM/randomness. Blends Feature Engine output
  into 5 sub-scores (trend/momentum/volume/volatility/risk, each 0–100)
  via fixed formulas over named config weights/thresholds
  (`OPPORTUNITY_WEIGHT_*`, `RSI_SCORE_FLOOR/CEIL`, `TIMEFRAME_WEIGHTS`,
  etc. — see `.env.example` for the full list), then a weighted
  `opportunity_score` (0–100) from the 5 sub-scores.
- Every aggregation step (timeframe components → a timeframe's
  sub-score, per-timeframe scores → the blended sub-score, sub-scores →
  the final score) goes through one shared weighted-average helper that
  renormalizes among whatever inputs are actually available — a missing
  indicator never fabricates a 0, it's excluded and the remaining
  weights renormalize. Result is `None` only if literally nothing was
  computable for that symbol.
- `select_top_candidates`: filters not-held symbols to
  `opportunity_score >= MIN_OPPORTUNITY_SCORE` (default 60), sorts
  descending, keeps the top `TOP_N_CANDIDATES` (default 5, **configurable**).
  A held position whose recomputed score falls below
  `EXIT_SCORE_THRESHOLD` (default 40) becomes an LLM exit-validation
  candidate. The gap between the two thresholds is a deliberate
  hysteresis band — a symbol scoring in between is never simultaneously
  too-weak-to-enter and forced-to-exit-if-held.

### Signal Agent (`src/agents/signal_agent.py`) — LLM validation gate
- `validate_opportunity(opportunity_summary, strategy_prompt, context)`
  is the only entry point now (`context` is `"entry"` or `"exit"`).
  Receives a curated digest (symbol, opportunity_score, the 5 sub-scores,
  volatility label, support/resistance, ADX, volume spike, and —
  exit-context only — the held position's entry price/qty/unrealized
  PnL%) — **never raw candles, never the full multi-timeframe feature
  dump**.
- Asks the LLM to accept or reject, with reasoning/risks/expected
  duration/invalidation point; only `decision` and `reasoning` drive
  control flow, the full verdict is stored losslessly as jsonb
  (`opportunity_evaluations.llm_raw_response`).
- Fails **closed**: an unparseable response or every model in the
  fallback chain failing both resolve to `decision: "reject"` — the LLM
  is a gate now, not the primary decision-maker, so a broken gate must
  not silently let a trade through.
- `reasoning` is what gets persisted to `trades.reasoning_text` on an
  accepted entry/exit — same column, same dashboard rendering as before.

### Risk Manager Agent (`src/agents/risk_manager.py`)
- **Safety-critical — build and unit-test this first (build order step 6).**
- Enforces, in order: circuit-breaker state check → capital limit check
  → position sizing → daily target/loss bookkeeping.
- Position sizing (resolved decision, §9): **fixed % of `capital_to_use`
  per trade**, capped at `max_concurrent_positions` open positions
  simultaneously. Both are new **configurable** columns on
  `capital_config` (see §6): `position_size_pct` (default 10%),
  `max_concurrent_positions` (default 5). At the default settings, at
  most 50% of allocated capital is deployed at once — deliberate buffer,
  not a hard requirement.
- Owns the circuit-breaker: tracks realized PnL for the current IST
  trading day, flips `circuit_breaker_triggered` and instructs Execution
  Agent to flatten when `max_daily_loss` is breached.
- Only sizes a position once the Opportunity Scorer has ranked the
  symbol a top candidate AND the Signal Agent's LLM validation has
  accepted it (§3) — ranking/ties are the Opportunity Scorer's job
  (`opportunity_score`, descending), not this module's.
- Per-trade stop-loss/take-profit (`exit_reason()`): the active strategy
  version's `params_json.stop_loss_pct`/`take_profit_pct` (decimal
  fraction of entry price, e.g. `0.02` = 2%) are enforced against every
  open trade's live ticker price, independent of the LLM signal for that
  cycle — a hit closes the position immediately rather than waiting for
  the LLM to say "sell". Either key can be omitted to leave that side
  unenforced. See `orchestrator.run_risk_check()` and §5.
  **Not a guarantee, same caveat as the circuit breaker above**: CoinDCX's
  spot API has no exchange-side stop order (confirmed against their docs —
  `market_order`/`limit_order` only; stop-limit/take-profit exist solely
  in their margin product, which this bot deliberately doesn't use — see
  §1 spot-only). So this is enforced by polling, not by the exchange
  watching the price continuously, and GitHub Actions free-tier cron is
  best-effort — the actual gap between checks can exceed the nominal 5
  minutes under platform load, with no hard ceiling. A fast, large move
  can still realize a loss/gain past the configured percentage before the
  next check fires. Closing this gap for real needs an always-on poller
  (not cron-triggered), which is a deliberate scope decision not taken —
  see if this ever actually bites before reaching for it.

### Execution Agent (`src/agents/execution/`)
- Shared interface (`base.py`): `place_order`, `get_fill`, `flatten_all`.
- `paper.py` — `PaperExecutionAgent`: simulates fills against live
  orderbook data with **configurable** slippage/fee model (slippage = a
  configurable bps constant applied against best bid/ask). Fee model
  mirrors CoinDCX's actual charges: 0.5% trading fee on trade value on
  both sides, +18% GST on that fee, and an additional 1% TDS (Income
  Tax Act s.194S) on trade value on **sells only** (not levied on
  buys — no "transfer" of the asset on acquisition). All three
  constants live in `paper.py`, adjust if CoinDCX's fee schedule
  changes.
- `real.py` — `RealExecutionAgent`: calls CoinDCX's authenticated/signed
  order endpoints, adds the same 1% sell-side TDS on top of the
  exchange-reported `fee_amount` since TDS isn't a documented field on
  the order response (unverified against a real fill — see the module's
  own caveat below). Only ever instantiated by the orchestrator when
  `strategy_versions.promoted_to_real = true` for the active version —
  this gate lives in the orchestrator, not just the execution agent, so
  there's no path that reaches real order placement with an unpromoted
  strategy.

### Evolution/Learning Agent (`src/agents/evolution_agent.py`)
- Runs nightly (separate GH Actions workflow, §7).
- Computes win rate, avg win/loss, drawdown from the day's (and trailing
  window's) trades per mode.
- Asks the LLM to propose an updated strategy prompt/params; result is
  saved as a new row in `strategy_versions` (never mutates an existing
  version — versions are immutable once created).
- Checks promotion criteria (§2) for the current paper version and sets
  `promoted_to_real` when met.
- Also runs the Learning Engine's periodic passes (§3a): `compute_feature_importance`
  and `generate_recommendations` — piggybacked on this existing nightly
  cron rather than a new workflow, since both are batch statistical
  passes over a growing dataset, not per-cycle work.

### Reporting Agent (`src/agents/reporting_agent.py`)
- Generates an HTML report covering both modes side by side: PnL vs
  target, trade log, current strategy version + changelog, model
  fallback stats (from `model_usage`), and a Learning Insights section
  (§3a) — best/worst regimes, symbols, score ranges, most-profitable
  hour/weekday, longest win/loss streak.

### Orchestrator (`src/orchestrator.py`)
- Single script invocation, one full cycle, then exit. Invoked per mode
  (paper, real) — real invocation is a no-op if no version is promoted.
- Sequence: check circuit-breaker state for today → stop-loss/take-profit
  sweep (`_sweep_stop_loss_take_profit`, zero LLM, also updates MFE/MAE
  for every open trade — §3a) → circuit-breaker recheck → Data Agent →
  **Pass 1** (pure, no LLM): Feature Engine + Opportunity Scorer score
  every scanned symbol (now including a `market_regime` classification),
  split into not-held/held, `select_top_candidates` picks the entry
  candidate set → **Pass 2**: for each scanned symbol, circuit-breaker
  check (same position as before every buy, preserved from the original
  design) → entry candidates: `find_similar_trades` (§3a) first, its
  result feeds both the Signal Agent's prompt and, after the LLM
  responds, `calibrate_confidence` blends the LLM's own confidence with
  the historical figure — a `MIN_FINAL_CONFIDENCE` gate (default 0,
  permissive) sits alongside the existing Risk Manager check before
  `place_order` → score-deteriorated held positions go straight to Signal
  Agent exit validation (no similarity search on exits — the SL/TP sweep
  and `EXIT_SCORE_THRESHOLD` already cover that path) → every symbol
  reaching Pass 2 gets exactly one `opportunity_evaluations` row logged
  regardless of outcome (most with `llm_decision = null` — the checkable
  proof LLM call volume actually dropped), plus `trades` / `daily_pnl` /
  `agent_logs` / `model_usage` as before → `process_closed_trades` (§3a)
  runs once at the end, catching up self-evaluation/statistics for any
  trade closed since the last pass, regardless of which path closed it.

## 3a. Trade Memory + Learning Engine (`src/learning/`)

Pure statistics over closed trades — no ML/RL, no external libraries.
Every completed trade's entry-time context is captured (`trades` +
`opportunity_evaluations`, linked via `opportunity_evaluations.trade_id`)
so later cycles can learn from it. Outputs are only as reliable as trade
volume allows — every ratio/correlation below returns `None` (or skips
writing entirely) rather than fabricating a number from too little data;
this is expected to be noisy until real trade history accumulates.

- **`statistics.py`**: `compute_bucket_statistics` extends
  `evolution_agent.compute_metrics` (win_rate/avg_win/avg_loss/
  cumulative_pnl/max_drawdown_pct — imported, not reimplemented) with
  Sharpe (`mean/stdev` of `pnl/capital_to_use` per trade), Sortino
  (deviation from zero over *all* trades, not the stdev of losses alone —
  a different, non-standard statistic), Calmar
  (`cumulative_pnl_pct / max_drawdown_pct`, both percent-normalized),
  expectancy, and profit factor. `process_closed_trades(mode)` is the
  path-independent catch-up entry point: finds closed trades without a
  `trade_evaluations` row yet (Python-side diff, not a DB join — no
  precedent for embedded Supabase queries in this codebase), regardless
  of whether they closed via the SL/TP sweep, an LLM-validated exit, or a
  circuit-breaker flatten, self-evaluates each, and upserts every
  `learning_statistics` bucket (symbol / market_regime /
  opportunity_score_bucket / confidence_bucket / strategy_version /
  weekday / hour — IST-converted) it belongs to, bounded by
  `LEARNING_HISTORY_WINDOW_DAYS`.
- **`trade_memory.py`**: `find_similar_trades` — Euclidean distance over
  the 5 already-computed sub-scores (not raw candles) against a bounded,
  time-windowed pool of past entries with known outcomes, filtered by
  `SIMILARITY_MAX_DISTANCE` then requiring `MIN_SIMILAR_TRADES` survivors
  before returning a historical win rate at all.
- **`confidence_calibration.py`**: `calibrate_confidence` blends the
  LLM's own stated confidence (new `confidence` field in
  `validate_opportunity`'s JSON contract) with the historical win rate,
  configurable weights, collapsing to AI-only when history is thin.
- **`feature_importance.py`**: point-biserial correlation (hand-rolled
  sums — `statistics.correlation` needs Python 3.10+, this repo's local
  dev interpreter is 3.9) between each raw Feature Engine value (primary
  timeframe) and win/loss outcome, gated behind a minimum sample size.
- **`recommendations.py`**: advisory-only threshold suggestions (e.g.
  "trades scoring ≥82 outperform the current `MIN_OPPORTUNITY_SCORE=60`
  by X%") — never auto-applied to config, human approval required, no
  dashboard surface yet (inspect the `recommendations` table directly).
  Idempotent by construction: skipped if not materially different from
  the latest existing recommendation for that metric.
- **`reports.py`**: `generate_learning_report_html`, wired into
  `reporting_agent.py`'s existing report as one more section, not a
  parallel report.
- Market regime classification lives in `src/features/opportunity_scorer.py`
  (`classify_market_regime`, folded into `score_opportunity`'s return —
  reuses `score_trend`, already computed) rather than a separate module:
  `sideways` / `high_volatility` / `strong_bull` / `weak_bull` /
  `strong_bear` / `weak_bear`, derived from ADX + trend_score, no separate
  "trending" label (redundant with strong_bull/strong_bear).

## 4. LLM integration (Groq default, Ollama Cloud alternative)

- Provider is **configurable**: `LLM_PROVIDER=groq` (default) or `ollama`
  (Ollama Cloud — `https://ollama.com`, authenticated via `OLLAMA_API_KEY`,
  not a local instance). Same retry/fallback/logging behavior either way;
  `src/groq_client.py`'s `chat()` is the single entry point both agents call,
  so switching providers is an env var change, not a code change.
- Model chain is **configurable** per provider (env var, ordered list — Groq
  deprecates models periodically): default
  `GROQ_MODEL_CHAIN=openai/gpt-oss-120b,qwen/qwen3.6-27b`,
  `OLLAMA_MODEL_CHAIN=gpt-oss:120b` (no `-cloud` suffix — that's only for
  routing through a local Ollama daemon, not this direct-to-`ollama.com` setup).
- On 429 or any API error: retry with exponential backoff on the current
  model, then fall back to the next model in the chain.
- Every call (success or failure, every model tried) logs to
  `model_usage`: model name, fallback_reason (null on first-try
  success), latency_ms, success.
- Verify during build (step 3) by forcing a failure (e.g. bad API key
  swapped in temporarily, or a monkeypatched 429) to confirm the
  fallback chain actually triggers and logs correctly — don't just trust
  the retry logic unexercised.

## 5. Deployment (free tier only)

- Trading cycle: GitHub Actions workflow, `cron: '*/10 * * * *'`,
  invokes the orchestrator script once per mode and exits. **Known
  limitation**: GitHub Actions cron is best-effort, not guaranteed —
  under platform load, runs can be delayed several minutes. The
  Risk Manager's daily bookkeeping must tolerate skipped/late cycles
  (it recomputes from `trades`/`daily_pnl`, not from cycle count).
- Risk check: separate `risk_check.yml` workflow, `cron: '*/5 * * * *'`
  (5 min is GitHub Actions' shortest supported schedule interval —
  going tighter isn't possible on the free tier, and per-cycle LLM cost
  is why the full trading cycle above stays at 10 min). Runs
  `orchestrator.py --risk-only`: stop-loss/take-profit + circuit-breaker
  sweep only, no LLM call and no market snapshot, so it's cheap enough
  to run twice as often as the signal cycle. This is what actually
  bounds how long a bad move can run unwatched — not the signal cycle's
  interval, since exits no longer wait on the LLM to notice.
- Evolution job: separate daily GH Actions workflow.
- Database: Supabase free tier (Postgres).
- Secrets: GitHub encrypted secrets for CI; `.env` (gitignored) for
  local dev. Never committed — `.env.example` documents the required
  keys with placeholder values.

## 6. Database schema (Postgres / Supabase)

```sql
-- capital_config: one row per mode ('paper' | 'real')
capital_config (
  mode                    text primary key,      -- 'paper' | 'real'
  total_capital           numeric not null,       -- paper: sim capital; real: full wallet balance (informational)
  capital_to_use          numeric not null,        -- amount Risk Manager is allowed to size against
  daily_profit_target     numeric not null,
  max_daily_loss          numeric not null,
  position_size_pct       numeric not null default 10,   -- % of capital_to_use per trade
  max_concurrent_positions int not null default 5,
  paused                  boolean not null default false, -- dashboard Start/Stop; orchestrator.py no-ops before any model/exchange call when true (0003_pause_flag.sql)
  updated_at              timestamptz not null default now()
)

-- strategy_versions: immutable once created
strategy_versions (
  id                 bigserial primary key,
  version_number     int not null,
  prompt_text        text not null,
  params_json        jsonb not null default '{}',
  promoted_to_real   boolean not null default false,
  notes              text,
  created_at         timestamptz not null default now()
)

-- trades: one row per position, paper or real
trades (
  id                  bigserial primary key,
  mode                text not null,              -- 'paper' | 'real'
  version_id          bigint not null references strategy_versions(id),
  symbol              text not null,              -- e.g. 'BTCINR'
  side                text not null,              -- 'buy' | 'sell'
  qty                 numeric not null,
  entry_price         numeric not null,
  exit_price          numeric,
  pnl                 numeric,
  fees                numeric not null default 0,
  status              text not null,              -- 'open' | 'closed' | 'flattened'
  opened_at           timestamptz not null default now(),
  closed_at           timestamptz,
  reasoning_text      text,                       -- full LLM reasoning for this trade
  stop_loss_price     numeric,                    -- entry-time dollar level, 0005_learning_engine.sql
  take_profit_price   numeric,
  entry_slippage_pct  numeric,                    -- signed, vs. market["last_price"] at decision time
  mfe_pct             numeric not null default 0, -- running max favorable excursion, updated by the SL/TP sweep
  mae_pct             numeric not null default 0, -- running max adverse excursion
  exit_reason         text,                       -- 'stop_loss' | 'take_profit' | 'ai_exit' | 'circuit_breaker'
  market_regime       text                        -- entry-time classification, see §3a
)

-- daily_pnl: one row per (date, mode)
daily_pnl (
  date                      date not null,
  mode                      text not null,
  realized_pnl              numeric not null default 0,
  trades_count              int not null default 0,
  target_hit                boolean not null default false,
  circuit_breaker_triggered boolean not null default false,
  primary key (date, mode)
)

-- agent_logs: structured log per agent call
agent_logs (
  id               bigserial primary key,
  timestamp        timestamptz not null default now(),
  agent_name       text not null,
  level            text not null,            -- 'info' | 'warning' | 'error'
  message          text not null,
  raw_llm_response jsonb
)

-- model_usage: every Groq call, including fallbacks
model_usage (
  id             bigserial primary key,
  timestamp      timestamptz not null default now(),
  model_used     text not null,
  fallback_reason text,                       -- null if first-try success
  latency_ms     int not null,
  success        boolean not null
)

-- opportunity_evaluations: one row per scanned symbol per cycle,
-- logged regardless of outcome (0004_opportunity_evaluations.sql)
opportunity_evaluations (
  id                  bigserial primary key,
  timestamp           timestamptz not null default now(),
  mode                text not null,
  symbol              text not null,
  version_id          bigint not null references strategy_versions(id),
  features            jsonb not null,          -- compute_multi_timeframe_features() output
  trend_score         numeric,
  momentum_score      numeric,
  volume_score        numeric,
  volatility_score    numeric,
  risk_score          numeric,
  opportunity_score   numeric,
  llm_decision        text,                    -- 'accept' | 'reject' | null (null = never reached LLM)
  llm_reasoning       text,
  llm_raw_response    jsonb,                   -- full parsed verdict, null when no LLM call
  risk_manager_result text,                    -- 'size' | 'block_circuit_breaker' | 'block_max_positions' | 'block_capital_limit' | null
  final_decision      text not null,           -- 'buy' | 'sell' | 'hold' | 'circuit_breaker'
  reason              text,
  trade_id            bigint references trades(id)  -- 0005_learning_engine.sql; one trade can have 2 rows (entry + an LLM-validated exit)
)

-- learning_statistics: EAV-style bucketed stats, upserted in place on
-- recompute (0005_learning_engine.sql, see §3a)
learning_statistics (
  id                      bigserial primary key,
  mode                    text not null,
  dimension_type          text not null,        -- symbol | market_regime | opportunity_score_bucket | confidence_bucket | strategy_version | weekday | hour
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
)

-- feature_importance: point-biserial correlation, primary timeframe only
feature_importance (
  id                bigserial primary key,
  mode              text not null,
  feature_name      text not null,             -- a Feature Engine FEATURE_KEYS entry
  correlation_score numeric,
  sample_count      int not null default 0,
  computed_at       timestamptz not null default now(),
  unique (mode, feature_name)
)

-- confidence_calibration: audit log, one row per entry-validation call
-- (not aggregate stats — what was actually applied to a specific decision)
confidence_calibration (
  id                        bigserial primary key,
  opportunity_evaluation_id bigint not null references opportunity_evaluations(id),
  ai_confidence             numeric,
  historical_confidence     numeric,
  ai_weight                 numeric,
  historical_weight         numeric,
  final_confidence          numeric,
  similar_trades_count      int not null default 0,
  created_at                timestamptz not null default now()
)

-- recommendations: advisory only, human approval required, never
-- auto-applied. Append-only (idempotency enforced in application code).
recommendations (
  id                bigserial primary key,
  mode              text not null,
  metric_name       text not null,
  current_value     numeric,
  recommended_value numeric,
  rationale         text,
  sample_size       int not null default 0,
  status            text not null default 'pending',  -- 'pending' | 'reviewed' | 'dismissed'
  created_at        timestamptz not null default now()
)

-- trade_evaluations: 1:1 self-evaluation child of trades
trade_evaluations (
  trade_id                      bigint primary key references trades(id),
  predicted_confidence          numeric,
  predicted_opportunity_score   numeric,
  actual_outcome_won            boolean not null,
  confidence_was_accurate       boolean,
  opportunity_score_was_accurate boolean,
  risk_assessment               text,           -- 'appropriate' | 'too_aggressive'
  stop_loss_assessment          text,           -- 'appropriate' | 'too_tight' | null
  target_assessment             text,           -- 'realistic' | 'too_ambitious' | null
  evaluated_at                  timestamptz not null default now()
)
```

No separate `market_regimes` table — `learning_statistics WHERE
dimension_type='market_regime'` already is that data; a second table
would duplicate it under a different name.

- Daily rollover boundary for `daily_pnl` and circuit-breaker state is
  **midnight IST** (Asia/Kolkata) — resolved decision, §9.
- Row-level security: dashboard's Supabase anon key is read-only across
  all tables; writes go through the Python agents using the service key
  (server-side / CI only, never shipped to the browser).

## 7. Deployment topology

```
GitHub Actions (cron */10) ─┬─> orchestrator.py --mode=paper ─┐
                             └─> orchestrator.py --mode=real  ─┤
                                 (no-op if nothing promoted)   │
                                                                ▼
GitHub Actions (daily cron) ──> evolution_agent.py ──┐    Supabase (Postgres)
                                                       └──────────▲
                                                                  │
Vercel (Next.js dashboard) ──── Supabase JS client (anon, RLS) ──┘
```

## 8. Dashboard (Next.js on Vercel)

- Supabase JS client directly from the frontend (read-only anon key +
  RLS) — no separate backend server.
- Screens: Overview (mode toggle, today's PnL vs target, circuit-breaker
  status, capital in use), Trades table, Evolution tracker (version
  history + performance chart), Config panel (edit capital/target/loss
  limit — authenticated), Model health (fallback stats chart).
- Charts: Recharts.
- Deferred to build order step 10 — not scaffolded yet (see §10).

## 9. Open decisions resolved (asked up front, answers locked in)

| Question | Decision |
|---|---|
| Daily reset timezone | IST (Asia/Kolkata), midnight rollover |
| Circuit-breaker action on trigger | Flatten all open positions immediately, then block new entries |
| Position sizing rule | Fixed % of `capital_to_use` per trade, capped at `max_concurrent_positions` |
| Initial symbol scope | Dynamic top 10 INR pairs by 24h volume each cycle, not a fixed list |

## 10. Build order

1. Repo scaffold, `.env` handling, Supabase schema + migrations
2. CoinDCX client (public market data first, verify live INR data pulls)
3. Groq wrapper with fallback chain (test by forcing a failure to confirm fallback triggers)
4. Database layer (models, read/write trades, config)
5. Paper trading cycle end-to-end, top-10-by-volume selection
6. Risk Manager rules with unit tests (capital limit, daily target, circuit breaker, position sizing) — do not skip
7. Evolution agent + versioning, run against a few days of paper data
8. Reporting agent (HTML report for both modes)
9. GitHub Actions workflows for the cron cycle and nightly evolution job
10. Next.js dashboard on Vercel wired to Supabase
11. Real trading Execution Agent, gated behind `promoted_to_real`, only after user review of paper history

**Current status: all 11 steps built.** `RealExecutionAgent` (step 11)
exists and is wired into the orchestrator, but is inert in practice —
`promoted_to_real` requires ≥`PROMOTION_MIN_PAPER_DAYS` (default 14) of
paper history, positive cumulative PnL, and drawdown under
`PROMOTION_MAX_DRAWDOWN_PCT`, so it won't fire until paper trading has
actually earned it. Its order-placement/fill-parsing is unverified
against a live fill (account balance was under CoinDCX's ₹100
min_notional at build time) — confirm with one small real order once
funds exist, before trusting it beyond the promotion gate.

## 11. Repo structure (scaffolded, step 1)

```
.
├── PROJECT_SPEC.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── config.py                  # env var loading (added when step 2 starts)
│   ├── orchestrator.py
│   ├── agents/
│   │   ├── data_agent.py
│   │   ├── signal_agent.py          # LLM validation gate, not primary decision-maker
│   │   ├── risk_manager.py
│   │   ├── evolution_agent.py
│   │   ├── reporting_agent.py
│   │   └── execution/
│   │       ├── base.py
│   │       ├── paper.py
│   │       └── real.py
│   ├── features/                    # deterministic, zero-LLM scoring pipeline
│   │   ├── feature_engine.py
│   │   └── opportunity_scorer.py    # includes classify_market_regime
│   ├── learning/                    # Trade Memory + Learning Engine, see §3a
│   │   ├── statistics.py
│   │   ├── trade_memory.py
│   │   ├── confidence_calibration.py
│   │   ├── feature_importance.py
│   │   ├── recommendations.py
│   │   └── reports.py
│   └── db/
│       ├── models.py
│       └── migrations/
│           ├── 0001_init.sql
│           ├── 0002_rls.sql
│           ├── 0003_pause_flag.sql
│           ├── 0004_opportunity_evaluations.sql
│           └── 0005_learning_engine.sql
├── tests/
│   └── test_risk_manager.py
├── dashboard/                      # Next.js app — deferred to step 10
└── .github/workflows/              # deferred to step 9
```

Only directories + `__init__.py` + the real migration SQL are created in
this commit. Agent modules stay unwritten until the plan below is
confirmed — no stub files with fake logic to rewrite later.
