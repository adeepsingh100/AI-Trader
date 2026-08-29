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
  `strategy_versions.promoted_to_real = true`. `evolution_agent.py`'s
  hourly `run_evolution()` delegates the actual decision to
  `src/learning/promotion_gate.py::evaluate_promotion()` — a
  multi-dimensional `PROMOTE`/`REJECT`/`EXTEND_VALIDATION` decision, not a
  boolean — and on `PROMOTE` sets `strategy_versions.promotion_eligible =
  true` and auto-flips `promoted_to_real` in the same run via
  `models.promote_version()`. Auto-promotion itself isn't new (it
  survived the original 3-raw-threshold version this replaced, §3e); what
  changed is the bar: sample-size floors (backtest/paper/walk-forward/
  TRUE paired-observation counts, not just elapsed days), risk/
  statistical/Monte-Carlo gates, regime/symbol robustness, an overfitting
  verdict, and a statistically significant, same-market-data improvement
  over the current real-mode champion — not 3-5 simple thresholds. The
  champion-vs-challenger significance test itself is PAIRED and is the
  ONLY test — no fallback to a weaker unpaired test ever exists (the old
  `src.backtest.strategy_comparison.compare` z-test is not imported or
  called anywhere in `promotion_gate.py`; it stays live only for its own
  different, genuine consumer, `simulation.py`'s exit-params candidate
  gate). Observations are matched by their shared decision-cycle
  identifier — each backtest-replay snapshot's `snapshot_time` — via
  explicit set intersection, never by list index/position, and a
  duplicate identifier on either side is dropped as ambiguous rather than
  silently resolved. Two DIFFERENT counts, never conflated:
  `paired_snapshot_count` (matched snapshot pairs) and
  `paired_return_observations` (one fewer — the consecutive-snapshot
  return deltas the statistical test actually consumes), gated
  independently by `PROMOTION_MIN_PAIRED_SNAPSHOTS` /
  `PROMOTION_MIN_PAIRED_RETURN_OBSERVATIONS` (never
  `min(champion_trades, challenger_trades)` — independent trade counts
  don't imply matched market observations). Significance itself is a
  **Moving Block Bootstrap** (`statistical_validation.moving_block_
  bootstrap_probability`, block length `PROMOTION_BOOTSTRAP_BLOCK_LENGTH`)
  over the paired return-difference series — resamples contiguous BLOCKS,
  not individual points, preserving the local temporal dependence a plain
  point-wise bootstrap would destroy — gated at `PROMOTION_MIN_
  CONFIDENCE_PCT` against an explicitly-named `bootstrap_probability_
  candidate_better_pct` statistic (the percentage of bootstrap resamples
  whose cumulative candidate-minus-champion difference is positive —
  never called generic "confidence"), deterministic given the same
  data/block_length/iterations/seed. A bot's first-ever promotion (no
  champion) marks the champion-comparison gate AND both paired sample
  gates `NOT_APPLICABLE` rather than leaving them permanently unresolved
  — otherwise every first promotion would deadlock at `EXTEND_VALIDATION`
  forever regardless of how clean every other gate looked. Missing
  required evidence (e.g. no historical candles ingested yet for the
  walk-forward/paired-observation backtest-replay gates) always yields
  `EXTEND_VALIDATION`, never a silent skip and never a promotion on
  partial evidence — see `promotion_gate.py`'s own module docstring for
  the full gate-by-gate breakdown and §3e for how it composes almost
  entirely from primitives §3e/§3c already built. Every evaluation
  (`PROMOTE`/`REJECT`/`EXTEND_VALIDATION` alike, not just a promotion) is
  written to `promotion_audit` (§6) — "no promotion may occur without a
  complete audit record" is satisfied trivially if only promotions were
  logged, so rejections/pending evaluations are too. A structural safety
  property survives regardless: promotion metrics are scoped to
  `get_closed_trades(mode, version["id"])` — that specific version's own
  trade history — so a freshly created version always starts its own
  `PROMOTION_MIN_PAPER_DAYS`/sample-size clock at zero regardless of how
  often `run_evolution()` re-checks it. A `PROMOTION_COOLDOWN_DAYS`
  (default 7) floor also blocks rapid-fire re-promotion regardless of how
  many candidates happen to clear every other gate in one run.
- **Automatic Rollback**: if a promoted real-mode champion's live
  performance later degrades, `strategy_health.py`'s existing nightly-now-
  hourly auto-suspend (§3d) already causes `get_latest_promoted_version()`
  to naturally fall back to the next-most-recent still-active promoted
  version (it excludes suspended rows) — this was always structurally
  true. What's new: when the suspended version IS the current real-mode
  champion, that fallback is now explicitly audited as a `promotion_audit`
  row (`event_type='rollback'`, recording which version was reinstated).
  No new monitoring machinery — `run_strategy_health(mode="real")` reuses
  the identical health computation already used for paper, just scoped to
  the champion's own real trades (its paper trades often stop growing once
  evolution moves on to a newer paper candidate, so real trades are the
  only reliable ongoing signal for it specifically).

## 3. Multi-agent architecture

All agents are separate, independently testable Python modules under
`src/agents/` (plus `src/features/` for the deterministic scoring
pipeline). One full cycle runs to completion and exits — no long-running
process (see §7):

```
Data Agent → Feature Engine → Opportunity Scorer → Candidate Filter
    → Risk Manager → Execution Agent → log
```

**Fully quant, zero LLM calls in this cycle**: a deterministic scorer
(zero AI, zero randomness) ranks every scanned symbol, and clearing
`MIN_OPPORTUNITY_SCORE`/`TOP_N_CANDIDATES` (or, for a held position,
dropping below `EXIT_SCORE_THRESHOLD`) **is** the trade decision — no
separate validation step asks anything for a second opinion (§4 on why,
and where an LLM is still used elsewhere, offline from this cycle).

### Data Agent (`src/agents/data_agent.py`)
- Pulls CoinDCX's public market endpoints (ticker, candles).
- Each cycle: fetches 24h ticker volume for all INR pairs, selects the
  **top 10 by volume** (Resolved decision — see §9: dynamic
  top-10-by-volume, not a fixed pair list), pulls candles for each of
  `FEATURE_TIMEFRAMES` (**configurable**, default `1m,15m,1h,1d` —
  CoinDCX's candles API only accepts `1m`/`15m`/`1h`/`1d`, others 422) at
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
  `opportunity_score` (0–100) from the 5 sub-scores. The "risk" sub-score
  is, despite the name, resistance headroom only (distance from the next
  resistance level) — kept as-is (DB column/`OPPORTUNITY_WEIGHT_RISK`/
  dashboard would need a migration to rename) but never treated as a
  stand-in for actual trade risk. Real risk (execution cost/liquidity,
  stop-distance, portfolio exposure, drawdown) is measured and gated
  independently — Net Expectancy Gate, risk-based position sizing,
  Portfolio Intelligence concentration caps, circuit breaker/promotion
  drawdown gate respectively (see Trade decisions and Risk Manager Agent
  below) — not blended into this one number.
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
  `EXIT_SCORE_THRESHOLD` (default 40) closes. The gap between the two
  thresholds is a deliberate hysteresis band — a symbol scoring in
  between is never simultaneously too-weak-to-enter and
  forced-to-exit-if-held.

### Trade decisions — fully quant, zero LLM calls (`src/orchestrator.py`)
- There is no LLM validation gate between the Opportunity Scorer and a
  trade. Reaching `MIN_OPPORTUNITY_SCORE`/`TOP_N_CANDIDATES` (entry) or
  dropping below `EXIT_SCORE_THRESHOLD` (exit) is necessary but not
  sufficient — an entry candidate must also clear the **Net Expectancy
  Gate** (`risk_manager.compute_net_expectancy_pct`) before an order is
  placed. Both gates are deterministic, no network call, no token budget,
  no parse failure ever in the path that opens or closes a position. (The
  module that used to sit here, `src/agents/signal_agent.py`, an LLM
  accept/reject gate, was removed entirely — see §4 for why.)
- **Net Expectancy Gate**: fees (`TRADING_FEE_PCT`+`GST_PCT_ON_FEE`, both
  legs, via `src/agents/execution/paper.py::fees()` reused directly so
  this is the exact fee the trade would actually pay) + TDS
  (`SELL_TDS_PCT`, sell leg) + spread (`EXPECTANCY_SPREAD_BPS`) +
  slippage (`SLIPPAGE_BPS`) netted against the resolved stop/target
  (below) and the system's own calibrated win-probability estimate
  (`calibrate_confidence`'s `final_confidence`, falling back to
  `opportunity_score` when there's no historical blend yet — never a new
  probability estimator). `net_expectancy_pct <= 0` is itself the "no
  trade" decision for **real mode**, unconditionally — code is allowed to
  do nothing. Pure percentage-of-notional math, no qty/entry_price needed,
  since every cost scales linearly with notional. **Paper mode** trades
  through a negative `net_expectancy_pct` by default
  (`PAPER_TRADES_ON_NEGATIVE_EXPECTANCY`, default true) — it risks
  nothing real, and gating it identically to real mode meant it could
  never accumulate the trade history confidence calibration and
  promotion both need in the first place. Still requires a resolvable
  stop/target (the gate itself, not just its sign) and still goes through
  every other check (risk sizing, exposure, circuit breaker) unchanged.
- **Stop-loss/take-profit resolution** (`risk_manager.resolve_exit_params`):
  the active strategy version's evidence-validated `params_json.
  stop_loss_pct`/`take_profit_pct` (already cleared the walk-forward/
  bootstrap/fitness gate, §3b/§3e) wins when configured. A leg
  `params_json` doesn't configure falls back to an ATR-derived value
  (`atr_pct * STOP_LOSS_ATR_MULTIPLIER`/`TAKE_PROFIT_ATR_MULTIPLIER`,
  clamped to `EXIT_PARAM_SWEEP_MIN_PCT`/`MAX_PCT`) — never both at once
  for the same leg, and never overriding a configured value. Closes the
  "no stop configured = unbounded downside" gap `statistics.py::
  _assess_risk` already self-flags as `"too_aggressive"`.
- **Risk-based position sizing**: `risk_manager.evaluate`'s existing flat/
  dynamic capital-fraction qty gets one additional cap —
  `(capital_to_use * RISK_PER_TRADE_PCT / 100) / (stop_loss_pct *
  last_price)` — so no single trade risks more than `RISK_PER_TRADE_PCT`
  of capital if its stop is hit. Strictly additive: `min()` with the
  existing formula, so it can only shrink qty, never grow it past what
  flat/dynamic sizing already allows.
- The confidence that gates position sizing (`calibrate_confidence`,
  `MIN_FINAL_CONFIDENCE`) now blends the scorer's own `opportunity_score`
  with historical win-rate on similar past trades, instead of an LLM's
  self-rating — same function, same weights
  (`CONFIDENCE_AI_WEIGHT`/`CONFIDENCE_HISTORICAL_WEIGHT`), just a
  quant-sourced input.
- `reasoning_text` on a trade, and `opportunity_evaluations.llm_reasoning`,
  are now a deterministic string built from the sub-scores and the
  threshold that was cleared — same columns, same dashboard rendering as
  before, just quant-authored instead of LLM-authored.
- An LLM is still consulted elsewhere in the codebase — see §4 — but only
  once an hour, offline from live trading, and only to *propose* a
  candidate value that still has to clear the same statistical gate as
  every other strategy change before it can matter.

### Risk Manager Agent (`src/agents/risk_manager.py`)
- **Safety-critical — build and unit-test this first (build order step 6).**
- Enforces, in order: circuit-breaker state check → capital limit check
  → position sizing → daily target/loss bookkeeping.
- Position sizing (resolved decision, §9): **fixed % of `capital_to_use`
  per trade** (or the Capital Allocation Engine's dynamic multiplier,
  §3d — `capital_config.sizing_mode`), capped at `max_concurrent_positions`
  open positions simultaneously. Both are new **configurable** columns on
  `capital_config` (see §6): `position_size_pct` (default 10%),
  `max_concurrent_positions` (default 5). At the default settings, at
  most 50% of allocated capital is deployed at once — deliberate buffer,
  not a hard requirement. On top of that, whichever formula ran gets one
  additional **risk-based cap** derived from stop distance
  (`RISK_PER_TRADE_PCT` of capital per trade, evaluate()'s optional
  `stop_loss_pct` kwarg) — `min()` with the existing qty, never a
  replacement, so it can only shrink size, never grow it.
- Owns the circuit-breaker: tracks realized PnL for the current IST
  trading day, flips `circuit_breaker_triggered` and instructs Execution
  Agent to flatten when `max_daily_loss` is breached.
- Only sizes a position once the Opportunity Scorer has ranked the
  symbol a top candidate (§3) — ranking/ties are the Opportunity Scorer's
  job (`opportunity_score`, descending), not this module's.
- Per-trade stop-loss/take-profit (`exit_reason()`): the active strategy
  version's `params_json.stop_loss_pct`/`take_profit_pct` (decimal
  fraction of entry price, e.g. `0.02` = 2%) are enforced against every
  open trade's live ticker price every cycle — a hit closes the position
  immediately, not deferred to any other check, and this live-recomputed
  check retroactively protects already-open positions when `params_json`
  changes. When a leg isn't configured in `params_json`, `exit_reason()`
  falls back to that specific trade's own frozen-at-entry stored
  `stop_loss_price`/`take_profit_price` (set at open time via
  `resolve_exit_params`'s ATR fallback — see Trade decisions above) rather
  than leaving that side unenforced — every position always has a defined
  risk boundary. See `orchestrator.run_risk_check()` and §5.
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
- Runs hourly (separate GH Actions workflow, §7).
- **Promotion monitor only** since the Scientific Strategy Optimization
  Framework (§3e) — no LLM call, no new `strategy_versions` row.
  Computes win rate, avg win/loss, drawdown, and a blended fitness score
  (§3e) from the current version's trades, then delegates the actual
  `PROMOTE`/`REJECT`/`EXTEND_VALIDATION` decision to
  `src/learning/promotion_gate.py::evaluate_promotion()` (§2/§3e) — sets
  `strategy_versions.promotion_eligible` and, on `PROMOTE`, auto-flips
  `promoted_to_real` in the same run, no human click. Also runs
  `src/db/models.py::purge_old_data()` (Data Retention, §3d) each pass,
  piggybacked on this already-hourly step rather than a new cron job.
- Previously also asked an LLM to freely rewrite the strategy's
  prompt_text/params_json every night and auto-promoted the instant 3
  simple thresholds cleared with no statistical check at all — retired
  entirely (§3e explains why). `strategy_versions` still exists;
  `params_json` is what live trading actually reads every cycle
  (`risk_manager.py`'s stop-loss/take-profit) — `prompt_text` is unread
  since the LLM validation gate that used it, `signal_agent.py`, was
  later removed entirely too (§4). Strategy evolution (including
  `stop_loss_pct`/`take_profit_pct`, previously only ever LLM-guessed)
  happens exclusively through the Adaptive Strategy Intelligence
  Engine's candidate pipeline (§3b/§3e) now.

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
  design) → entry candidates: reaching this branch already means the
  quant scorer accepted the candidate — `find_similar_trades` (§3a) feeds
  `calibrate_confidence`, which blends `opportunity_score` with the
  historical win-rate figure (no LLM confidence, no LLM call anywhere in
  this path) behind the `MIN_FINAL_CONFIDENCE` gate (default 0,
  permissive) → `resolve_exit_params` resolves the effective stop/target
  → the **Net Expectancy Gate** (`compute_net_expectancy_pct`) must clear
  before the Risk Manager's `evaluate()` (capital + risk-based-sizing +
  concentration checks) and `place_order` → score-deteriorated held
  positions close unconditionally on `EXIT_SCORE_THRESHOLD`, no
  similarity search or extra validation (the SL/TP sweep already covers
  that path) → every symbol reaching Pass 2 gets exactly one
  `opportunity_evaluations` row logged regardless of outcome (`reason`
  distinguishes `not_a_candidate`/`confidence gated`/`net_expectancy
  gated`/a Risk Manager block/an actual fill), plus `trades` / `daily_pnl`
  / `agent_logs` as before → `process_closed_trades` (§3a) runs once at
  the end, catching up self-evaluation/statistics for any trade closed
  since the last pass, regardless of which path closed it.

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
  of whether they closed via the SL/TP sweep, a score-drop exit, or a
  circuit-breaker flatten, self-evaluates each, and upserts every
  `learning_statistics` bucket (symbol / market_regime /
  opportunity_score_bucket / confidence_bucket / strategy_version /
  weekday / hour — IST-converted / exit_reason / rsi_bucket /
  stoch_rsi_bucket / atr_volatility_bucket) it belongs to, bounded by
  `LEARNING_HISTORY_WINDOW_DAYS`. The RSI/StochRSI/volatility buckets
  (Phases 7-9 of the strategy-refinement audit) read the entry-time
  indicator snapshot already stored on every `opportunity_evaluations`
  row (`features` jsonb) — no new data collection. RSI/StochRSI share the
  same fixed 0-100-scale edges (`<30, 30-40, ..., >80`); the volatility
  bucket is a 5-way evidence-only split (very_low/low/medium/high/extreme)
  distinct from the Feature Engine's 3-way live-scoring split, which is
  untouched. `recommendations.py::generate_indicator_bucket_recommendations`
  flags "avoid bucket X" via the same generic two-proportion z-test as the
  existing regime/symbol "avoid" recommendations — advisory only, never
  auto-applied, same `RECOMMENDATION_MIN_SAMPLE_SIZE` gate.
- **`trade_memory.py`**: `find_similar_trades` — Euclidean distance over
  the 5 already-computed sub-scores (not raw candles) against a bounded,
  time-windowed pool of past entries with known outcomes, filtered by
  `SIMILARITY_MAX_DISTANCE` then requiring `MIN_SIMILAR_TRADES` survivors
  before returning a historical win rate at all.
- **`confidence_calibration.py`**: `calibrate_confidence` blends a
  quant confidence signal (the Opportunity Scorer's own `opportunity_score`
  — there's no LLM in this path anymore, see §4) with the historical win
  rate, configurable weights, collapsing to score-only when history is
  thin.
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

## 3b. Adaptive Strategy Intelligence Engine (`src/learning/`)

Closes the loop from the Learning Engine's statistics into recommended
parameter changes — walk-forward validated and simulated before a
candidate is even created. Two trust levels, split by whether the
candidate's target is a DB row or a deployment env var:
**exit-params candidates** (`stop_loss_pct`/`take_profit_pct` — target
`strategy_versions.params_json`, a DB row) **auto-activate** into a new
`strategy_versions` row the moment they pass every statistical gate —
`simulation.py::_activate_exit_params_candidate`. **Everything else**
that could change what the bot trades (opportunity-scorer weights,
thresholds, avoid-symbol/avoid-regime — these target `OPPORTUNITY_WEIGHT_*`-
style `os.getenv()` constants in `config.py`, not a DB row) stays
advisory, human-approved in Supabase, same as `recommendations` already
worked before this — auto-applying those means rewriting deployment env
vars and forcing a redeploy, a materially bigger change than a DB write,
deliberately out of scope for now. The one other automatic piece is the
confidence modifier chain, an extension of the already-automatic (and
inert-by-default, `MIN_FINAL_CONFIDENCE=0`) `calibrate_confidence` gate.

- **`feature_importance.py`** (extended): `compute_feature_importance`
  now accepts a `timeframes` list (default `[PRIMARY_TIMEFRAME]`,
  unchanged behavior) — passing `FEATURE_TIMEFRAMES` computes correlation
  independently per timeframe using each trade's already-stored
  multi-timeframe feature dump, zero new candle fetches. New
  `compute_subscore_correlation_weights(mode)` correlates the 5
  already-flat opportunity_evaluations sub-score columns against
  win/loss, normalizes positive correlations into a weight distribution,
  and caches them in `feature_importance` (`timeframe="blended"`) — read
  back live (but cheaply, no recomputation) by
  `trade_memory._feature_importance_weights`, which is no longer a
  permanent no-op stub. New `score_separation_p_value(trades, weights)`
  recomputes each trade's opportunity score under a candidate weight set
  and z-tests whether winners and losers separate significantly better
  than under the current weights.
- **`statistics.py`** (extended): `streaks()` (moved from `reports.py`,
  now also returns the *current* streak, not just the longest, since the
  live confidence chain needs it every cycle) and stdlib-only two-sample
  z-test helpers (`z_test_two_proportions`, `z_test_two_means`, normal
  approximation via `math.erf`) — used for both a recommendation's stated
  confidence and its walk-forward pass/fail gate.
- **`recommendations.py`** (extended, `generate_recommendations` itself
  unchanged): `generate_weight_recommendations` (candidate
  `OPPORTUNITY_WEIGHT_*` values, accepted only if they separate
  winners/losers better than the live weights), `generate_regime_recommendations`
  ("avoid regime X" plus regime-conditioned weight recommendations),
  `generate_symbol_recommendations` ("avoid symbol X" plus a per-symbol
  optimal-threshold sweep for confidence/opportunity_score/stop_distance/
  volatility, via a generalized `_find_optimal_threshold` shared with the
  original sweep). Every generator: only ever re-evaluates trades already
  taken (no counterfactual discovery — that would need a real
  candle-replay backtester, out of scope), and is idempotent the same way
  the original threshold sweep already was.
- **`simulation.py`** (new): walk-forward validation — a recommendation
  is regenerated using only the older TRAIN fraction of
  `LEARNING_HISTORY_WINDOW_DAYS`, then evaluated only against the newer
  TEST fraction, never touched during generation. On a statistically
  significant pass (`SIGNIFICANCE_THRESHOLD`), lazily creates an
  `adaptive_strategy_versions` candidate row — only for proposals that
  clear the bar, so that table only ever holds genuine candidates.
  Currently simulates the mode-wide weight recommendation and the
  mode-wide `MIN_OPPORTUNITY_SCORE` threshold recommendation; regime-/
  symbol-scoped recommendations stay recommend-only (each already needs
  the full sample floor per bucket just to be generated once).
- **`adaptive_strategy_engine.py`**: `AdaptiveStrategyEngine.analyze(mode)`
  — the single composed entry point, calling every generator above plus
  the simulations, then logging a summary. Never executes a trade, never
  writes to `config.py` or any trading table. Runs as its own hourly
  step in `evolution.yml` (after `evolution_agent`, no new workflow). As
  of the Scientific Strategy Optimization Framework (§3e) this is the
  **sole** source of strategy-change candidates —
  `evolution_agent.py`'s own `generate_recommendations`/
  `compute_feature_importance` calls (previously duplicated here and
  there, kept only for a log line) were removed when its LLM-tuning loop
  was retired, not merged into this module.
- **Adaptive confidence chain** (`confidence_calibration.py`, extended):
  `calibrate_confidence` gains optional `regime_modifier`/
  `symbol_modifier`/`recent_performance_modifier` params — each a bounded
  adjustment (`BUCKET_MODIFIER_SENSITIVITY`/`BUCKET_MODIFIER_CAP` for
  regime/symbol, based on that bucket's win rate vs. the active version's
  overall win rate; `RECENT_STREAK_WIN_MODIFIER_CAP`/
  `RECENT_STREAK_LOSS_MODIFIER_CAP` for the current win/loss streak over
  the last `RECENT_PERFORMANCE_LOOKBACK_TRADES` closed trades), summed
  onto the existing AI+historical blend and clamped to 0-100. Computed
  once per cycle in `orchestrator.py` (not once per candidate), gated on
  each bucket clearing `RECOMMENDATION_MIN_SAMPLE_SIZE`. No live
  per-trade timeframe modifier — a trade blends across
  `FEATURE_TIMEFRAMES`, so there's no single per-trade timeframe bucket
  to look up the way there is for symbol/regime; that signal lives in
  `feature_importance`'s per-timeframe correlation instead.
- **`reports.py`** (extended): `generate_adaptive_strategy_report_html`,
  wired into `reporting_agent.py` as one more section — best/accepted/
  rejected recommendations, simulation results, candidate/approved
  adaptive strategy versions.

## 3c. Event-Driven Backtesting & Walk-Forward Validation Engine (`src/backtest/`)

Fills the gap §3a/§3b's own docstrings openly admit to: neither ever
replays candles or discovers a trade that wasn't actually taken — "a real
backtester (candle replay) would be a much bigger feature and isn't being
built here." This is that feature. Purely additive; `orchestrator.py`'s
live `run_cycle` is untouched. Confirmed live (via a direct API check, not
assumed) that CoinDCX's public candles endpoint accepts `startTime`/
`endTime` even though `coindcx_client.py`'s wrapper never exposes them —
real historical replay is possible, capped at 500 candles/call, walked
backward via pagination in `data_provider.py`. Candle `time` is bar-OPEN
time (the most-recent no-range-filter candle is always still forming),
which pins the no-look-ahead rule exactly: a bar is visible at simulated
time `t` only if `open_time + interval_duration <= t`.

**What's reused unchanged** (this is where "same trading logic in both
modes" is actually honored, not `run_cycle` itself — see below):
`feature_engine.compute_multi_timeframe_features`, `opportunity_scorer.
score_opportunity`/`select_top_candidates`, `risk_manager.evaluate`/
`exit_reason`/`circuit_breaker_triggered`, `execution/paper.py`'s `fees`
formula (made public), and `learning/statistics.py`'s Sharpe/Sortino/
Calmar/win-rate/profit-factor/expectancy/z-tests. **What's new**: the
event-reactor loop itself, because `run_cycle` is a live-polling shell
(Supabase writes and `get_ticker()` on every call) fundamentally
incompatible with "replay chronologically, no network calls in the hot
loop." Two cadences kept deliberately separate from the tick granularity:
`BACKTEST_DECISION_CYCLE_MINUTES` (mirrors `trading_cycle.yml`'s 10-min
cron) and `BACKTEST_RISK_CHECK_MINUTES` (mirrors `risk_check.yml`'s 5-min
cron) — ticking the decision pass on every candle would simulate a bot
checking 5-10x more often than the live one ever does.

- **`events.py`**: `MarketEvent/SignalEvent/OrderEvent/FillEvent/
  PositionEvent/PortfolioEvent/RiskEvent/TimeEvent` + `EventQueue`.
- **`simulation_clock.py`**: `is_bar_closed` is the no-look-ahead rule's
  single source of truth; `SimulationClock` derives its own day boundary
  for daily-PnL bucketing (never imports `risk_manager.today_ist()`, which
  is real wall-clock time).
- **`data_provider.py`**: paginated historical fetch (network, only ever
  called by `ingest_data.py`) + `CandleStore`, an in-memory, closed-bar-only
  reader loaded once per run — zero network/DB calls in the hot loop.
- **`order_manager.py` / `execution_simulator.py`**: `OrderType` (market/
  limit/stop/stop_limit/trailing_stop) + realistic fill simulation
  (spread/slippage/partial fills/rejections/expiry) — commission reuses
  `execution/paper.py`'s exact fee formula. Live only ever issues market
  orders (CoinDCX spot has no exchange-side resting order); the richer
  types are a real, generic capability, not something the default parity
  backtest of the current live strategy exercises. Fully in-memory —
  never imports `src.db.models` or `src.coindcx_client`, so a backtest run
  can never leak into live Supabase state.
- **`portfolio_manager.py`**: cash/equity/positions/realized+unrealized
  PnL/exposure, mark-to-market equity curve — genuinely new (the only
  existing drawdown, `evolution_agent._max_drawdown_pct`, walks the
  trade-pnl sequence, not an intraday equity curve). Spot-only like live —
  no margin/leverage machinery.
- **`engine.py`**: `BacktestEngine` — the reactor loop. Replicates all
  **three** circuit-breaker checkpoints `run_cycle` has (top-of-decision-
  pass, post-SL/TP-sweep, per-candidate). Symbol universe is an explicit,
  user-supplied list, never reconstructed from a live turnover ranking —
  CoinDCX has no historical ticker/turnover series to replay, so
  defaulting to "today's top-N over history" would be survivorship bias;
  every report says this. Entry/exit decisions are quant-only,
  deterministic, mirroring `orchestrator.py` exactly — there was
  previously an opt-in LLM signal-agent validation path here
  (`BACKTEST_USE_LLM_SIGNAL_AGENT`); it was removed along with
  `signal_agent.py` itself (§4), not just defaulted off, since live
  trading no longer has an LLM-validated path for a backtest to mirror.
  `ingest_data.py` (CLI) backfills `historical_candles` for
  `[start_date - BACKTEST_WARMUP_BUFFER_DAYS, end_date]`, never just the
  requested window — `FEATURE_CANDLE_LIMIT`/`EMA_TREND_PERIOD_4` need up
  to ~200 closed daily bars before scores stop being `None`, so without
  the buffer every run would silently find zero candidates for its first
  ~200 days.
- **`performance_analyzer.py`**: reuses `statistics.compute_bucket_statistics`
  directly; adds gross profit/loss, Omega ratio, Ulcer index, rolling
  Sharpe/volatility/drawdown, monthly/annual returns, exposure time,
  capital utilization. Recovery factor is computed at report time only,
  never stored — numerically identical to Calmar, same "don't store one
  fact under two names" precedent as §3b.
- **`trade_analysis.py`**: per-trade MFE/MAE/slippage/commission/return/
  risk-reward/exit-reason/confidence/opportunity-score/regime.
- **`walk_forward_validator.py`**: real rolling multi-fold validation
  (train window → test window, never overlapping, stepped forward) —
  distinct from and doesn't modify `simulation.py`'s existing single-split
  trade-repartition logic (different question: would this parameter set
  have made money on a historical period it never saw, vs. does a weight
  recommendation separate already-observed outcomes better).
- **`strategy_comparison.py`** — **ANALYTICS ONLY**, not the live
  auto-promotion authority: pairwise run comparison reusing
  `z_test_two_proportions`/`z_test_two_means` directly — "only recommend
  promotion if statistically superior" means the test rejects the null in
  B's favor, not just that B's raw number is bigger. Its sole runtime
  consumer is `simulation.py`'s exit-params CANDIDATE gate (whether an
  `adaptive_strategy_versions` row is even worth creating), an ordinary
  unpaired two-sample z-test over one backtest-replay window. `promotion_
  gate.py` never imports or calls it — the authoritative champion-vs-
  challenger statistic for real-money promotion is `promotion_gate.py`'s
  own paired Moving Block Bootstrap comparison (§3e), a different, later,
  time-series-aware method. Despite the field name, this module's
  `promotion_recommended` output never drives `strategy_versions.
  promoted_to_real` anywhere — only `evaluate_promotion()`'s decision does.
- **`statistical_validation.py`**: confidence intervals via **seeded
  bootstrap resampling**, not a parametric t-interval — this codebase has
  zero numpy/scipy, and a hand-rolled regularized-incomplete-beta
  implementation for a real t-CDF is real numerical bug surface for little
  gain over the existing z-test at backtest sample sizes; bootstrap needs
  no distributional assumption at all, arguably the more honest answer for
  small fold sizes. Plus Monte Carlo trade-order resampling (drawdown
  path-dependency) and a parameter-stability sweep. All randomness draws
  from a local `random.Random(BACKTEST_RANDOM_SEED)` instance, never the
  global `random` module — reruns are bit-identical.
- **`overfitting_detection.py`**: aggregates walk-forward fold results +
  parameter stability into a verdict (`robust`/`marginal`/`overfit`). This
  module only classifies; it never deletes anything and never touches
  `strategy_versions`/`promoted_to_real` itself — but the verdict IS a
  mandatory reject-capable gate inside the live auto-promotion pipeline
  (`promotion_gate.py`: `verdict == "overfit"` → `REJECT`), same as the
  candidate-pipeline's own `recommendations`/`adaptive_strategy_versions`
  status marking. Auto-promotion elsewhere in this codebase is fully
  automatic — no human-approval step anywhere.
- **`report.py`**: HTML (reusing `reporting_agent._table`) + CSV/JSON via
  stdlib. No PDF — the HTML report is already print-to-PDF-ready in any
  browser, and this codebase's zero-non-essential-dependency discipline
  (no numpy/scipy despite far heavier justification) argues against adding
  one just for PDF rendering. Standalone per-run artifact, not wired into
  `reporting_agent.py`'s live dashboard report.

No new GitHub Actions workflow — on-demand CLI (`python -m
src.backtest.engine`/`ingest_data`), same precedent as `seed_config.py`,
not a recurring job.

## 3d. Institutional Reliability Layer

Everything §3a-§3c built is trading intelligence; this is production
reliability on top of it — data quality, portfolio-aware risk, execution
quality, drift/health monitoring, observability, and fault tolerance.
Confirmed via research before building: none of it existed anywhere in
this repo, not even partially, except order types already simulating
correctly in `src/backtest/` (nothing chose between them yet) and LLM
calls already having retry/backoff/fallback (`groq_client.py`, nothing
else did). Architecturally different from §3a-§3c: those were purely
additive; several pieces here **must** touch the live trading hot path
(`risk_manager.py`, `orchestrator.py`, `coindcx_client.py`, `db/models.py`)
to mean anything — done narrowly, with every touched function keeping its
existing signature and default behavior.

**Market Data Quality Engine + Data Repair Engine** (`src/data_quality/`):
`validator.py`'s `MarketDataValidator` checks every candle for missing
bars/duplicates/negative or invalid OHLC/out-of-order timestamps/gaps/
zero-volume/extreme spikes/exchange outages/clock drift (live-fetch path
only)/symbol mismatch/timeframe changes, each mapped to a configurable
`ignore|warn|reject|quarantine` severity. `repair.py`'s `DataRepairEngine`
auto-fixes only what's safely fixable (small gaps ≤`DATA_REPAIR_MAX_GAP_BARS`
via linear interpolation, exact-duplicate merges, reordering) — a
reject/quarantine-severity issue is never silently repaired, and every
repair returns a logged entry (`data_quality_log`), never a silent
mutation. One shared entry point for both live (`data_agent.py`, right
after `get_candles()`) and backtest (`data_provider.py::ingest`, once at
ingest time) — not forked per pipeline.

**Portfolio Intelligence Engine** (`src/portfolio/intelligence.py`): pure
functions, no DB/network access — the caller supplies `positions` and a
`price_history` dict **already truncated to the caller's current point in
time** (the module never windows/slices beyond what it's handed, the fix
for a real look-ahead trap a design review caught: a careless backtest
call site could otherwise hand it a future-inclusive series "for
convenience"). Computes correlation matrix + rolling correlation, sector/
coin-category exposure (via a configurable `COIN_CATEGORY_MAP`, unmapped
symbols fall into `"uncategorized"` — no external crypto taxonomy exists
to query), stablecoin allocation, exchange exposure (always 100% —
single-exchange bot), max concentration, net/gross exposure (equal by
construction, spot-only/no shorting), beta (vs. `PORTFOLIO_BETA_PROXY_SYMBOL`,
default BTC), risk contribution, portfolio volatility, historical VaR/
Expected Shortfall (sorted-return percentile method, no distributional
assumption — same reasoning as §3c's bootstrap-over-parametric choice),
diversification score (1 − Herfindahl index). All hand-rolled stdlib math
(Python 3.9 compatible — deliberately not `statistics.covariance`/
`correlation`, 3.10+ only), same "no numpy/scipy" discipline as
`learning/statistics.py`'s z-tests.

Concentration caps scale with `capital_config.max_concurrent_positions`
(`100/max_concurrent_positions × MAX_POSITION_CONCENTRATION_MULT_OF_EQUAL_SHARE`)
rather than a fixed institutional-style percentage — a real bug an
integration test caught: a flat 25% cap blocked nearly every first trade
in this bot's actual 2-5-position range, since one position in a small
book is structurally 100% of it. `risk_manager.evaluate()` gained new
**optional** kwargs (`symbol`, `portfolio_positions`, `price_history`) —
omitted, behavior is byte-identical to before; supplied, a concentration
gate runs before the existing capital-limit check.

**Capital Allocation Engine** (`src/portfolio/capital_allocation.py`): the
highest-risk piece — replaces the flat `capital_to_use × position_size_pct
/ 100` formula with a multiplicative blend of independently configurable
factors (correlation/volatility/drawdown/exposure/strategy-performance/
regime/confidence, each clamped, the combined product clamped again).
**Rollout is paper-first, not automatic**: migration `0008` adds
`capital_config.sizing_mode text default 'flat'` — `'flat'` is today's
exact formula (byte-identical, verified by a regression test), `'dynamic'`
calls the new engine. Nothing in code flips this column; a human sets
paper's row to `'dynamic'` in Supabase after reviewing behavior, real's row
stays `'flat'` until they choose otherwise — the same "human edits a row
directly, no auto-promotion" invariant `paused`/`promoted_to_real` already
run on. The dynamic result still flows through the unchanged
`committed_capital(open_trades) + trade_capital > capital_to_use` ceiling —
the multiplier feeds that gate, never bypasses it.

**Execution Optimizer** (`src/execution_optimizer/optimizer.py`): pure
recommendation engine (MARKET vs. LIMIT, reusing `src/backtest/
order_manager.py`'s `OrderType` — no duplicate enum) estimating fill
probability/cost/delay/slippage from spread/liquidity/volatility/order-size/
recent-fill-rate. `RealExecutionAgent` stays fully untouched (market-only,
per its own documented unverified/inert status). `PaperExecutionAgent`
gained an optional, config-gated (`EXECUTION_OPTIMIZER_ENABLED`, default
false) path to act on a LIMIT recommendation same-cycle — modeled as
filling at a half-spread-improved price with probability
`estimated_fill_probability`, an explicit single-shot simplification since
paper trading has no cross-cycle resting-order infrastructure (that's what
`order_manager.py` is for, built for the backtest engine's multi-tick event
loop, not this synchronous per-cycle call).

**Feature Drift Detection** (`src/learning/drift_detection.py`): hand-rolled
Population Stability Index for feature-value distribution drift, a
correlation-magnitude delta for feature-importance trend, and
`z_test_two_proportions`-based (reused) win-rate/confidence-calibration/
opportunity-score-accuracy drift — baseline window vs. recent window, only
flags a *statistically significant worsening*, never an improvement or
noise. Runs as its own independent step in `evolution.yml` (never merged
into `evolution_agent.run_evolution()` or `adaptive_strategy_engine.py` —
the same "don't couple independent learning steps" rule those two already
follow). Writes to `drift_alerts` — advisory only, same as every other
`src/learning/` output.

**Strategy Health Engine** (`src/learning/strategy_health.py`): health
score (0-100, `Excellent/Good/Warning/Critical`) per `strategy_version`
from rolling Sharpe/drawdown/win-rate/profit-factor (`learning/statistics.py`,
reused), recent-vs-historical performance (z-tests, reused), and
walk-forward pass rate where a backtest exists for that version (omitted,
never fabricated, otherwise). Migration `0008` adds
`strategy_versions.status text default 'active'` — the same mutable-flag
pattern `promoted_to_real` already uses on this table. **A design review
caught a real silent-no-op risk here**: `get_latest_version()`/
`get_latest_promoted_version()` were unfiltered `ORDER BY version_number
DESC LIMIT 1` queries — both now filter `status != 'suspended'`, and
paper mode's prior hard crash on "no version" now distinguishes "never
bootstrapped" (still a crash) from "every version suspended" (a graceful
no-op, since a crash-loop every 10 minutes is a worse outcome). Suspension
is **status-only, never a delete** — reversible in Supabase at any time —
and only fires when `STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED` (default true)
and the trade count clears `RECOMMENDATION_MIN_SAMPLE_SIZE`.

**Production Monitoring + Self-Diagnostics** (`src/monitoring/`): scoped
to what's real for stateless ~10-minute GitHub Actions cron invocations
(Supabase as the only durable state), not invented long-running-server
metaphors. `metrics.py`'s `track()` context manager wraps timing/success
capture around `orchestrator.run_cycle`'s market-snapshot fetch, writing to
one generic `system_metrics` table (jsonb-bundle pattern, not N
single-purpose tables); `resource_snapshot()` is a stdlib `resource`/
`shutil.disk_usage` read of the runner's own process, the closest
meaningful thing to "CPU/memory/disk" for a script that runs seconds and
exits. `diagnostics.py`'s `run_health_check()` checks DB reachability,
market-feed freshness, learning-engine freshness, execution-engine
configuration, portfolio-position sanity, and recommendation-engine
reachability — added as a new step in `risk_check.yml` (already the
finest-grained cron, 5 min) rather than a 4th workflow file, which itself
risks silently going stale unnoticed.

**Audit System** — reuse first, not a new write path. `opportunity_evaluations`/
`confidence_calibration`/`trades` already capture every decision point in
`run_cycle` today (timestamp/component/input/decision/output/reason/
strategy-version/confidence/trade-id). `src/audit/trail.py::get_decision_trail()`
is a **read** function joining those three into one chronological timeline.
Only two genuinely missing fields became new columns on an existing table:
`opportunity_evaluations.config_version` (`trail.py::config_version()`, a
short hash of the live scoring/threshold constants, for reproducibility)
and `.market_regime` (already computed by the opportunity scorer, just not
persisted before). A design review flagged that layering new write calls
into the same hot loop being hardened by per-symbol isolation (below) would
work against that hardening — this sidesteps it entirely.

**Resilience** (`src/resilience.py`): `retry_with_backoff()` — the same
backoff shape `groq_client.py` already used for LLM calls, now the one
shared implementation — wraps every **read** call in `coindcx_client.py`
and the handful of hot-path reads in `db/models.py` that gate whether a
cycle can start at all (`get_capital_config`/`get_latest_version`/
`get_latest_promoted_version`/`get_daily_pnl`/`get_open_trades`).
`create_order` is deliberately **never** retried — a failed request whose
response was lost but which actually succeeded server-side would place a
second order on retry, a real double-submission risk a plain re-read
doesn't carry (the reason a full mechanical retry-every-`.execute()`-call
retrofit across all ~55 `db/models.py` functions was rejected in favor of
this narrower, correctness-checked set — writes get retried only where they
use `.upsert()`, which is naturally idempotent). A DB-backed circuit
breaker (`circuit_breaker_state` table: `coindcx_api`/`supabase`/`llm`,
consecutive-failure count, cooldown) survives across cron invocations and
fails **open** on its own write errors (a Supabase outage can't block
itself from being recorded as a Supabase outage). Per-symbol
`try/except` isolation was added to `orchestrator.run_cycle`'s Pass 2 loop
— a confirmed real gap: one symbol's exception used to abort every
remaining symbol in the cycle even though the cycle is otherwise safe to
retry from a clean DB-read state; this is the actual root-cause fix for
"crash recovery" in a stateless cron architecture, not new checkpointing
infrastructure (there's no persistent daemon to checkpoint).

**Data Retention** (`db/models.py::purge_old_data`): plain `DELETE ...
WHERE <column> < cutoff` on the highest-write-volume, non-permanent
tables — `opportunity_evaluations`/`confidence_calibration` (cutoff =
`LEARNING_HISTORY_WINDOW_DAYS`, reused rather than a second constant,
since nothing ever queries either table beyond that window anyway) and
`agent_logs`/`model_usage`/`system_metrics`/`data_quality_log` (cutoff =
the new `OPERATIONAL_LOG_RETENTION_DAYS`, default 30 — pure debug/ops
logs, never read past recent history). `trades`/`strategy_versions`/
`recommendations`/`adaptive_strategy_versions`/`strategy_simulations`/
`learning_statistics`/`feature_importance`/`drift_alerts`/
`strategy_health_scores`/`promotion_audit`/`historical_candles` are never
purged — the actual ledger, small-row-count decision history, compact
rollups, or
low-volume/valuable backtest data respectively. Runs every hour,
piggybacked on `evolution_agent.run_evolution()` (already scheduled, no
new cron) — fails open, a purge error never blocks the promotion-monitor
logic it shares a step with. Exists because `opportunity_evaluations` is
written every scanned symbol every cycle — the table that actually filled
the free-tier Supabase disk once already.

No new dependency of any kind — everything above is stdlib plus this
repo's own existing modules, matching the zero-numpy/scipy discipline the
codebase already prides itself on.

## 3e. Scientific Strategy Optimization Framework

Replaces the old "no trades → LLM guesses new prompt/params → auto-promote"
loop with a research pipeline: Trade Memory → Performance Analysis →
Weakness Detection → Hypothesis Generation → Candidate Strategy →
Statistical Validation (walk-forward, bootstrap CI, optional backtest
replay) → Promotion. Never modifies a strategy because trade count,
confidence, or win rate alone moved — every change is backed by
statistical evidence. Originally shipped with every step past Statistical
Validation gated on a human click (see the retirement note below); full
automation of the exit-params-candidate → strategy_versions and
promotion_eligible → promoted_to_real steps was reintroduced afterward —
§2 and §3b above cover exactly what's automatic now and why it's a
different, safer shape than what the retirement below describes.

**The retirement**: `evolution_agent.py::propose_next_version` sent the LLM
a bare `{current_prompt, current_params, metrics}` blob with an open-ended
"propose an improved prompt and params" instruction, and saved whatever
came back as a new `strategy_versions` row every single night, no
statistical gate — plus auto-flipped `promoted_to_real` the instant 3 raw
thresholds cleared, the only place in this codebase real money moved with
zero human approval. Both are deleted. `strategy_versions`/`prompt_text`
still exist (`params_json` is what live trading actually reads every
cycle — `prompt_text` is now unread since the LLM validation gate that
used it, `signal_agent.py`, was later removed entirely, see §4) but
`evolution_agent.run_evolution()`
shrinks to a promotion-readiness monitor (§2). Strategy evolution —
including `stop_loss_pct`/`take_profit_pct`, previously only ever
LLM-guessed — moved entirely into `adaptive_strategy_versions`' already-
rigorous candidate pipeline (§3b), extended rather than rebuilt:

- **`rejection_analysis.py`** (new): `rejection_breakdown(mode)` — ranks
  WHY candidates didn't trade ("38% blocked by concentration limit, 24%
  below MIN_OPPORTUNITY_SCORE, ..."), reading `opportunity_evaluations`
  rows `orchestrator.py` already writes for every scanned symbol every
  cycle (`final_decision="hold"`, `reason`/`risk_manager_result`) — fully
  captured data, previously never read back by anything.
- **`weakness_detection.py`** (new): `identify_weaknesses(mode)` — worst/
  best bucket per dimension (a thin ranking over `learning_statistics`,
  which already covers symbol/market_regime/opportunity_score_bucket/
  confidence_bucket/strategy_version/weekday/hour, plus a new
  `exit_reason` dimension) and worst/best indicator (over
  `feature_importance`'s existing point-biserial correlations).
- **`fitness.py`** (new): a configurable multi-objective blend — default
  30% profit factor / 25% Sharpe / 20% expectancy / 15% win rate / 10%
  drawdown penalty (`FITNESS_WEIGHT_*`), reusing
  `opportunity_scorer.weighted_average`'s renormalize-among-available
  convention so a missing component never skews the score. The 4 metric→
  0-100 component functions were previously private duplicates inside
  `strategy_health.py`; consolidated here as the one authoritative
  implementation, `strategy_health.py` imports them back.
- **`recommendations.py`** (extended): `generate_exit_params_recommendations`
  — sweeps candidate `stop_loss_pct`/`take_profit_pct` against expectancy
  using each trade's `mfe_pct`/`mae_pct` to approximate the outcome under a
  candidate distance (the same approximation `_assess_stop_loss`/
  `_assess_target` already made per-trade, generalized into a sweep — not
  a full price-path replay, a stated limitation). `generate_recommendations`
  gained an optional `weakness_context` param so a rationale can cite a
  corroborating weakness-detection finding as evidence.
  `generate_ai_exit_params_recommendations` (new) is an AI-assisted sibling
  — an LLM call (Groq, auto-falling back to Gemini; see §4) proposes a
  `stop_loss_pct`/`take_profit_pct` value from a digest of current params
  and baseline stats, written as an ordinary `recommendations` row
  (`category="exit_params"`) indistinguishable, downstream, from the
  pure-stat sweep's own output — `simulate_exit_params_recommendation`
  doesn't know or care which generator produced the row it's testing, so
  the AI's proposal clears the exact same walk-forward/bootstrap/fitness
  gate before it can matter. Any LLM failure here (quota, outage,
  unparseable response) just means no AI candidate that run — logged,
  never raised, never blocks the pure-stat sweep or the rest of
  `AdaptiveStrategyEngine.analyze()`.
- **`simulation.py`** (extended): every `simulate_*` function now also
  runs a bootstrap confidence-interval gate
  (`backtest.statistical_validation.bootstrap_confidence_interval`,
  previously orphaned — zero callers) after its z-test passes, requiring
  the CI's lower bound to still be positive (skipped for weight
  candidates, whose validation is win/loss *separation*, not a returns
  series, so a returns-CI doesn't apply). `simulate_exit_params_recommendation`
  additionally runs a real `BacktestEngine` replay — baseline vs candidate
  params over the same historical window, compared via
  `backtest.strategy_comparison.compare` (previously orphaned) — plus
  `backtest.walk_forward_validator.run_walk_forward` (previously
  orphaned) for multi-fold out-of-sample stability, whenever historical
  candle data exists for the traded symbols and a caller supplies a
  `symbol_to_pair` mapping (the one thing here that needs a network-
  derived value — `adaptive_strategy_engine.py` builds it best-effort,
  fail-open, from a public unauthenticated CoinDCX call; omitted, the
  function still runs its always-available checks). Every simulation now
  writes a `research_note` (Observation/Weakness/Hypothesis/Simulation/
  Walk Forward/Decision prose, not a changelog line) and a
  `validation_detail` jsonb bundle with the raw numbers, and a passing
  candidate's `adaptive_strategy_versions` row carries its `fitness_score`.
- **`promotion_gate.py`** (new): `evaluate_promotion()` — the multi-
  dimensional `PROMOTE`/`REJECT`/`EXTEND_VALIDATION` decision §2 describes,
  called from `evolution_agent.run_evolution()` in place of the old 5-gate
  boolean. Almost entirely composition over what this section already
  built: `fitness.py` (multi-objective score, unchanged), `bootstrap_
  confidence_interval` + a new `monte_carlo_drawdown_distribution`
  Monte-Carlo gate (probability-of-profit via a dedicated bootstrap-
  resample pass, catastrophic-drawdown probability from that function's
  now-exposed raw shuffled-distribution list — additive, existing callers
  destructure by key), `walk_forward_validator.run_walk_forward` +
  `overfitting_detection.detect` for the walk-forward/overfitting gate,
  a paired same-market-data comparison for champion-vs-challenger
  (candidate's and champion's `params_json` replayed via `BacktestEngine`
  over the IDENTICAL symbols/date range — "same market data" is only
  honest via backtest replay, since paper trades happen sequentially,
  never simultaneously; this is `_paired_champion_comparison`, evaluated
  via the Moving Block Bootstrap described below — never
  `strategy_comparison.compare`, which has zero coupling to this module,
  see its ANALYTICS ONLY note further down). New pieces genuinely added:
  sample-size floors (`PROMOTION_MIN_BACKTEST_TRADES`/`_WALK_FORWARD_TRADES`/
  `_PAPER_TRADES`/`_PAIRED_SNAPSHOTS`/`_PAIRED_RETURN_OBSERVATIONS`) that
  route to `EXTEND_VALIDATION` (never
  `REJECT` — "not enough evidence" isn't the same claim as "the evidence
  says no"); regime/symbol robustness (candidate's own bucketed
  `learning_statistics`-shaped stats compared against the champion's same
  bucket, plus a Herfindahl-style symbol-profit-concentration check); a
  complexity-delta (`params_json` keys changed vs champion) feeding the
  score's simplicity component; a weighted Promotion Score (§2's weight
  list) that's necessary but never sufficient — every hard gate above
  must independently pass regardless of score; a `PROMOTION_COOLDOWN_DAYS`
  floor. Every evaluation writes a `promotion_audit` row (§6) via
  `evolution_agent.py`, not this module directly (kept a pure decision
  engine, no DB writes of its own beyond the read-only lookups its gates
  need). `strategy_health.py`'s existing auto-suspend gained one
  conditional write of the same kind — a `promotion_audit` `'rollback'`
  row when the version it just suspended was the live real-mode champion
  (Automatic Rollback, §2) — fails open, same as the module's other
  advisory writes. Hardening pass on top of the above (same 3-way decision,
  same full automation): the backtest trade-count sample-size gate
  (`PROMOTION_MIN_BACKTEST_TRADES`) is now actually enforced (previously
  configured but never checked); a missing Sharpe improvement is missing
  evidence (`EXTEND_VALIDATION`), never silently a pass; the champion-vs-
  challenger significance test is the paired comparison described in §2,
  not candidate-alone profitability; `execution_quality` in the Promotion
  Score is real per-trade `entry_slippage_pct` data scored against
  `SLIPPAGE_BPS` (or `None` + reweighted among the rest), never a neutral
  50 placeholder. Second hardening pass, closing the remaining loopholes:
  the paired comparison is now the ONLY significance test (the unpaired
  fallback for when it couldn't be computed is gone — missing paired
  evidence is `EXTEND_VALIDATION`, never a substitute pass); observations
  are matched by their shared `snapshot_time` identifier via explicit
  intersection, never by list index; `PROMOTION_MIN_PAIRED_OBSERVATIONS`
  gates the TRUE matched count (retires `PROMOTION_MIN_CHAMPION_
  CHALLENGER_TRADES`'s `min(champion_trades, challenger_trades)` proxy);
  the statistic is explicitly named `bootstrap_probability_candidate_
  better_pct`; and a bot's first-ever promotion marks both the champion-
  improvement gate and the paired-observations sample gate
  `NOT_APPLICABLE` so it can still reach `PROMOTE` (previously the sample
  gate stayed permanently unresolved with no champion to pair against,
  deadlocking every first promotion at `EXTEND_VALIDATION`). Third
  hardening pass, statistical rigor: the significance test is now a
  **Moving Block Bootstrap** (`statistical_validation.moving_block_
  bootstrap_probability`) over the paired return-difference series
  instead of an ordinary point-wise bootstrap — financial returns are
  time-dependent, and resampling individual points destroys that local
  dependence; deterministic given the same data/`PROMOTION_BOOTSTRAP_
  BLOCK_LENGTH`/iterations/seed. `PROMOTION_MIN_PAIRED_OBSERVATIONS` is
  retired and split into `PROMOTION_MIN_PAIRED_SNAPSHOTS` (matched
  snapshot pairs) and `PROMOTION_MIN_PAIRED_RETURN_OBSERVATIONS` (one
  fewer — the actual return series the statistical gate consumes), never
  conflated. A duplicate `snapshot_time` on either engine's own side is
  now dropped as ambiguous before matching, not silently kept. The old
  `strategy_comparison.compare` z-test — already informational-only after
  the second pass — is no longer imported or called by this module at
  all; it remains exactly as it was for its own genuine, different
  consumer (`simulation.py`'s exit-params candidate backtest-replay
  gate), just structurally unreachable from the promotion pipeline. One
  authoritative result dict (mean/median/std/p25/p75/p95 difference,
  bootstrap method/block_length/iterations/seed, confidence threshold, an
  explicit PASS/FAIL status+reason) is computed once and consumed
  verbatim everywhere it's needed — nothing recomputes significance.
- **`statistics.py`** (extended): `accuracy_rates(trade_ids)` — aggregate
  confidence/opportunity-score/risk/stop-loss/target accuracy percentages
  over `trade_evaluations`, which `_evaluate_trade` already tagged
  per-trade; previously only ever consumed by `drift_detection.py`'s
  baseline-vs-recent alerting, never rolled up as a plain report metric.
  Average R-multiple stayed out of scope — not every trade has a
  well-defined initial risk (`stop_loss_pct` optional), and a shaky
  approximation would be worse than the honest gap.
- **`adaptive_strategy_engine.py`** is now the sole authoritative
  strategy-change pipeline (see the §3b update above) — its `analyze()`
  wires weakness detection and rejection analysis in before generating
  recommendations, and validates exit-params candidates with the network-
  optional `symbol_to_pair` built above.
- **`reports.py`** (extended): the adaptive-strategy HTML section gained
  Weaknesses Found, Rejection Breakdown, and Fitness columns on the
  simulation/version tables — additive to the existing report, not a
  parallel one.

Migration `0010_scientific_optimization.sql`: `strategy_versions.
promotion_eligible` (§2), `strategy_simulations.research_note`/
`.validation_detail`, `adaptive_strategy_versions.fitness_score` — all
nullable/safe-defaulted, zero behavior change until the corresponding code
ships. Migration `0011_promotion_audit.sql` (Promotion Gate, above): new
`promotion_audit` table (§6) — additive only, no column changes to any
existing table. No new dependency in either — stdlib plus this repo's own
modules throughout.

### Evidence-Driven Learning Progression (bootstrap-learning)

A single flat `RECOMMENDATION_MIN_SAMPLE_SIZE=20` gated the *overall*
"enough evidence to attempt this at all" check in every generator/
simulator — correct in isolation, but on a fresh paper-trading history it
meant the whole pipeline stayed silent (`[]` everywhere, no explanation)
until 20 trades existed, then jumped straight to full validation rigor
with no visibility into what was happening in between. First fix (staged,
still trade-count-only thresholds) still ignored everything
`orchestrator.py` already logs besides closed trades — rejected
candidates, symbols/regimes/hours/features/confidence seen. Second pass:
a new `EvidenceEngine` measures ALL of that, and `LearningStatus` becomes
the single authority every gate asks (`status.can_generate_hypotheses()`
etc.) instead of comparing a raw trade count inline.

`RECOMMENDATION_MIN_SAMPLE_SIZE` (still 20) keeps its original, narrower
job unchanged throughout: a per-bucket/per-subset credibility floor
(train/test halves, per-regime/per-symbol buckets, per-feature-pair
counts) — a different, still-valid concern from "is there enough evidence
overall," never touched by any of this.

**The load-bearing design decision**: evidence-readiness substitutes for
trade count only where the underlying capability doesn't need trade
OUTCOMES. Rejection analysis and coverage reporting don't need a single
closed trade, so **BOOTSTRAP→OBSERVATION is fully evidence-driven** — ANY
one of several dimensions clearing its bar unlocks it. Hypothesis
generation (z-test on win/loss separation), simulation (train/test split
of outcomes), and candidate validation (bootstrap CI on trade PnLs) are
statistically irreducible — no amount of "20 symbols seen" makes a
win/loss z-test valid on a sample it wasn't valid on before. Relaxing
those via coverage would be lowering statistical standards through a side
door, so **OBSERVATION→HYPOTHESIS→SIMULATION→VALIDATION stay gated on
trade count alone**, unchanged numbers, unchanged rigor:

| Stage | Unlocks via | Gate |
|---|---|---|
| BOOTSTRAP→OBSERVATION | data collection is already unconditional (`orchestrator.py` logs every trade/rejection/feature/score/confidence regardless of stage); OBSERVATION itself unlocks on ANY of: 25 closed trades, 500 rejected opportunities, 25 trading hours covered, 20 symbols covered, 6 market regimes covered, 80% feature coverage, or 40% blended evidence readiness | `EvidenceEngine` + `EVIDENCE_*` constants (`src/config.py`) |
| OBSERVATION→HYPOTHESIS | 100 closed trades (statistically irreducible — trade-count only) | `LEARNING_STAGE_HYPOTHESIS_MIN_TRADES` |
| HYPOTHESIS→SIMULATION | 250 closed trades | `LEARNING_STAGE_SIMULATION_MIN_TRADES` |
| SIMULATION→VALIDATION (candidate creation) | 500 closed trades | `LEARNING_STAGE_VALIDATION_MIN_TRADES` |
| (independent) | feature importance (`feature_importance.compute_feature_importance`) unlocks at 50 trades — a distinct, deliberately different floor from HYPOTHESIS's 100, not one of the 5 `can_*()` capabilities | `LEARNING_FEATURE_IMPORTANCE_MIN_TRADES` |

A simulation can pass its z-test/bootstrap-CI validation at the SIMULATION
stage without yet being allowed to create a candidate —
`simulation.py::_create_candidate_version` checks
`status.can_create_candidate()` independently of the statistical
pass/fail; the `strategy_simulations` row still honestly records
`passed=true` (that column's meaning is unchanged), and its `research_note`
gains a "Stage gate: ..." line explaining the deferral so it's never
ambiguous why a passing simulation produced no candidate.

**`src/learning/evidence_engine.py`** (new): `EvidenceEngine.collect(mode)`
measures — never changes a strategy, purely a read-side pass — closed/
winning/losing trades, rejected/candidate opportunities, symbols/market-
regimes/trading-hours covered, feature coverage (fraction of
`feature_engine.FEATURE_KEYS` with a correlation on record), confidence
coverage (fraction of scanned candidates that reached LLM scoring at
all), learning coverage (fraction of the 8 `learning_statistics`
dimension types populated), plus two Root Cause Analysis extensions:
`symbols_rarely_qualifying` (seen often, rejected often, never bought) and
`regimes_with_no_candidates` (regime seen on scanned candidates, never on
a buy). Reuses 4 already-existing model queries — zero new DB functions.
`compute_evidence_readiness(evidence)` blends 7 of those dimensions into
one 0-100% score via `opportunity_scorer.weighted_average` (the same
renormalize-among-available primitive `fitness.py` already reuses),
weights configurable via `EVIDENCE_WEIGHT_*` (`src/config.py`).
Deliberately out of scope: "which feature combinations consistently
fail" — needs multivariate analysis over rejected candidates' feature
vectors, a real, separate feature, not folded in here.

**`src/learning/learning_status.py`**: `compute_learning_status(mode)` now
returns a `LearningStatus` dataclass (not a plain dict) with five capability
methods — `can_generate_hypotheses()`, `can_simulate()`, `can_validate()`,
`can_create_candidate()` (delegates to `can_validate()` — candidate rows
are validation's output, not a separately-thresholded gate, one number
behind both names), and `can_promote()` (reads the `promotion_eligible`
flag `src.learning.promotion_gate.evaluate_promotion()` already computes
with its own full rigor — sample sizes, risk/statistical/Monte-Carlo
gates, regime/symbol robustness, champion improvement, promotion score —
never recomputes promotion logic here). `recommendations.py`'s 5 generators and
`simulation.py`'s 3 simulators each gained an optional
`status: LearningStatus | None = None` param (same threading pattern
`generate_recommendations`'s existing `weakness_context` param already
used) — `None` means "compute it myself" (every function stays
independently callable), but `adaptive_strategy_engine.analyze()` computes
one `LearningStatus` and threads the same instance into all 8 calls,
avoiding 8x redundant `EvidenceEngine` passes per run. Every
`if len(closed) < CONSTANT: return []` this replaced is gone — each
generator/simulator now asks `status.can_generate_hypotheses()` /
`status.can_simulate()` instead.

Consumed by `evolution_agent.run_evolution()` and
`adaptive_strategy_engine.analyze()` (added to their return dicts and log
lines — purely additive, neither engine's own logic changed),
`reports.py`'s HTML report (new "Learning status" + "Evidence coverage
breakdown" sections, first, above "Weaknesses found"), and mirrored in the
dashboard's `/learning` page (Evidence Readiness % is now the page's
headline stat, with a coverage breakdown grid). No new table — every field
is derived on read from tables that already exist, same "don't store a
derivable fact under a second name" precedent as everywhere else in this
codebase.

## 3f. Multi-Strategy-Type Support

Before this, `strategy_versions` had no `mode` column at all — one single
global lineage, differentiated only by `promoted_to_real` and which trades
reference which `version_id`. `strategy_type` is a new dimension,
orthogonal to `mode`, letting genuinely different strategies (e.g. today's
short-term "default" plus a longer-horizon "swing") run concurrently with
independent capital, circuit breaker, and learning evidence — never
blended into one confused signal.

**`src/config.py::STRATEGY_PROFILES`**: a small fixed Python registry, not
a dynamic/DB-driven system. `"default"` reuses the pre-existing bare env
vars byte-for-byte (proves zero behavior change); `"swing"` reads its own
`SWING_*`-prefixed env vars — timeframe weights leaning on 1h/1d only
(CoinDCX's candles API only supports `{1m,15m,1h,1d}`, confirmed — "swing"
means reweighting that fixed set, not fetching new timeframes), ~2x wider
ATR-fallback stop/target multipliers, a looser `exit_score_threshold` so
it tolerates pullbacks. A profile only governs live scoring weights and
the ATR *fallback* — the evidence-validated `params_json` value on a
strategy_versions row (per strategy_type lineage) always wins once one
exists, same as before. Adding a 3rd type later is one new dict entry, no
other code changes.

**Ships dormant.** A strategy_type only ever runs once its `capital_config`
row exists (`models.get_active_strategy_types(mode)`, intersected with
`STRATEGY_PROFILES` at every call site) — activating one is a
`seed_config.py` run (it now also prompts for `strategy_type`), not a
deploy. `default`'s capital/behavior is untouched until a second type is
deliberately activated.

**Schema** (migration `0014_multi_strategy_types.sql`): `strategy_type`
added to `strategy_versions`, `promotion_audit`, `recommendations`,
`strategy_simulations`, `adaptive_strategy_versions`; `capital_config`'s
PK becomes `(mode, strategy_type)`, `daily_pnl`'s becomes `(date, mode,
strategy_type)` (independent circuit breaker per type);
`learning_statistics`/`feature_importance`'s unique constraints widen to
include it (required, not optional — without this, one type's stats
`ON CONFLICT`-overwrite the other's bucket row). No column on `trades` or
`opportunity_evaluations` — both already carry `version_id →
strategy_versions.strategy_type`, so it's derivable via join, same
no-duplicate-derivable-fact precedent as everywhere else in this schema.

**Orchestrator**: `run_cycle`/`run_risk_check` loop every active
strategy_type, each with its own independent scoring/entry/exit pass —
but the market snapshot (candle fetch) and per-symbol feature computation
happen ONCE per cycle and are shared across every active type's scoring
pass (only the weighting/thresholds differ per type, never the raw
candles), and are skipped entirely if every active type turns out
paused/tripped that cycle (no wasted API call). `flatten_all` and every
DB read/write below the top-level loop take `strategy_type` too, so
tripping one type's circuit breaker never flattens another's positions.

**Evolution/adaptive engine**: `evolution_agent.run_evolution`,
`adaptive_strategy_engine.analyze`, and `drift_detection.run_drift_detection`
all loop the same way, returning `{strategy_type: {...}}`. Each type gets
its own independent promotion cooldown clock and champion slot — `default`
and `swing` can each independently reach real-mode promotion and end up
trading real money concurrently, each in its own capital sleeve, with no
special-casing beyond the scoping already described.

## 4. LLM integration (Groq, auto-falling back to Gemini — hourly, offline from live trading)

**Not in the live trade path at all.** `src/agents/signal_agent.py`, the
per-trade LLM accept/reject gate, was removed entirely — every entry/exit
is now a deterministic function of the Opportunity Scorer's already-computed
score (§3, "Trade decisions — fully quant, zero LLM calls"). This followed
a real incident: at up to `TOP_N_CANDIDATES` LLM calls every ~10-minute
cycle, Groq's free-tier per-model daily token quota was exhausted fast
enough to cause a multi-day trading outage before anyone noticed — a
provider issue blocking live trading is the specific failure mode this
architecture now makes structurally impossible, not just less likely.

The LLM's sole remaining role is **hourly**, inside
`AdaptiveStrategyEngine.analyze()` (§3b/§3e):
`generate_ai_exit_params_recommendations` asks it to propose a
`stop_loss_pct`/`take_profit_pct` candidate from recent trade
statistics — one call per run, ~24/day, comfortably inside Groq's free
quota even before Gemini fallback. That proposal is never applied
directly: it's written as an ordinary `recommendations` row and has to
clear the same walk-forward/bootstrap-CI/fitness-floor gate as every
other candidate (§3e) before it can become a live `adaptive_strategy_versions`
candidate. A failed or unavailable LLM call here just means no AI
candidate that hour, logged and skipped — never a blocked run, since this
step is offline from live trading entirely.

- No provider-select env var — every call tries the full Groq chain
  first, then automatically falls through to the full Gemini chain if
  every Groq model fails. `src/groq_client.py`'s `chat()` is the single
  entry point; callers never know which provider actually answered.
- Model chain is **configurable** per provider (env var, ordered list):
  default `GROQ_MODEL_CHAIN=openai/gpt-oss-120b,qwen/qwen3.6-27b`,
  `GEMINI_MODEL_CHAIN=gemini-2.5-flash`. Gemini's default reasoning
  ("thinking") pass is explicitly disabled (`thinkingConfig.
  thinkingBudget: 0`) — left on, it burns the whole output-token budget
  on its `<think>` chain before ever emitting the requested JSON, so
  every call came back truncated and unparseable until this was found
  and fixed.
- On 429 or any API error: retry with exponential backoff on the current
  model, then fall back to the next model in the chain; when the entire
  Groq chain is exhausted, fall through to the Gemini chain the same way.
- Every call (success or failure, every model tried, either provider)
  logs to `model_usage`: model name, fallback_reason (null on first-try
  success), latency_ms, success.

## 5. Deployment (free tier only)

- Trading cycle: GitHub Actions workflow, `cron: '*/10 * * * *'`,
  invokes the orchestrator script once per mode and exits. **Known
  limitation**: GitHub Actions cron is best-effort, not guaranteed —
  under platform load, runs can be delayed several minutes. The
  Risk Manager's daily bookkeeping must tolerate skipped/late cycles
  (it recomputes from `trades`/`daily_pnl`, not from cycle count).
- Risk check: separate `risk_check.yml` workflow, `cron: '*/5 * * * *'`
  (5 min is GitHub Actions' shortest supported schedule interval — going
  tighter isn't possible on the free tier). Runs `orchestrator.py
  --risk-only`: stop-loss/take-profit + circuit-breaker sweep only, no
  market snapshot, so it's cheap enough to run twice as often as the
  signal cycle. This is what actually bounds how long a bad move can run
  unwatched, not the signal cycle's interval.
- Evolution job: separate hourly GH Actions workflow (§3b/§4) — cheap
  even at this cadence, since every gate downstream is keyed off
  accumulated trade data/elapsed calendar time, not run frequency.
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

-- feature_importance: point-biserial correlation. timeframe added in
-- 0006_adaptive_strategy_engine.sql — raw Feature Engine keys use their
-- own timeframe, the 5 opportunity-scorer sub-scores always use the
-- explicit sentinel 'blended' (see §3b)
feature_importance (
  id                bigserial primary key,
  mode              text not null,
  feature_name      text not null,             -- a Feature Engine FEATURE_KEYS entry, or a sub-score name (trend_score, ...)
  timeframe         text not null,             -- '1m' | '15m' | '1h' | '1d' | 'blended'
  correlation_score numeric,
  sample_count      int not null default 0,
  computed_at       timestamptz not null default now(),
  unique (mode, feature_name, timeframe)
)

-- confidence_calibration: audit log, one row per entry-validation call
-- (not aggregate stats — what was actually applied to a specific decision).
-- *_modifier columns added in 0006_adaptive_strategy_engine.sql (§3b's
-- adaptive confidence chain) so every stage, not just final_confidence,
-- is individually auditable.
confidence_calibration (
  id                        bigserial primary key,
  opportunity_evaluation_id bigint not null references opportunity_evaluations(id),
  ai_confidence             numeric,
  historical_confidence     numeric,
  ai_weight                 numeric,
  historical_weight         numeric,
  final_confidence          numeric,
  similar_trades_count      int not null default 0,
  regime_modifier           numeric,
  symbol_modifier           numeric,
  recent_performance_modifier numeric,
  created_at                timestamptz not null default now()
)

-- recommendations: advisory only, human approval required, never
-- auto-applied. Append-only (idempotency enforced in application code).
-- category/confidence/evidence/batch_id added in
-- 0006_adaptive_strategy_engine.sql (§3b) so weight/regime/symbol
-- recommendations share this table rather than duplicating it.
recommendations (
  id                bigserial primary key,
  mode              text not null,
  metric_name       text not null,
  current_value     numeric,
  recommended_value numeric,
  rationale         text,
  sample_size       int not null default 0,
  status            text not null default 'pending',  -- 'pending' | 'reviewed' | 'approved' | 'dismissed'
  category          text not null default 'threshold', -- 'threshold' | 'weight' | 'regime' | 'symbol'
  confidence        numeric,                    -- (1 - p_value) * 100 from the walk-forward z-test
  evidence          jsonb,                       -- supporting trade ids / affected bucket refs, variable per category
  batch_id          uuid,                        -- groups co-generated rows (e.g. the 5 weight recommendations from one call)
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

-- strategy_simulations: walk-forward train/test result for one
-- recommendation batch (0006_adaptive_strategy_engine.sql, §3b) — this
-- table IS "Walk Forward Results" too, same computation reported
-- together, not a second artifact.
strategy_simulations (
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
)

-- adaptive_strategy_versions: versions QUANTITATIVE PARAMETERS (a
-- params_json snapshot of tunable adaptive constants) — orthogonal to
-- strategy_versions above, which versions LLM PROMPT TEXT. Created
-- lazily, only for simulations that pass. Exit-params candidates
-- auto-activate into a new strategy_versions row the same run
-- (simulation.py::_activate_exit_params_candidate) — no separate
-- "currently active" table needed there, get_latest_version() answers
-- it. Weight/regime/symbol candidates still target env vars, not a DB
-- row, so those stay advisory — a human still copies approved values
-- into env vars by hand for those.
adaptive_strategy_versions (
  id                             bigserial primary key,
  mode                           text not null,
  version_number                 int not null,
  params_json                    jsonb not null,
  source_recommendation_batch_id uuid,
  source_simulation_id           bigint references strategy_simulations(id),
  status                         text not null default 'candidate',  -- 'candidate' | 'approved' | 'rolled_back'
  notes                          text,
  created_at                     timestamptz not null default now()
)
```

No separate `market_regimes` table — `learning_statistics WHERE
dimension_type='market_regime'` already is that data; a second table
would duplicate it under a different name. Likewise no separate "Adaptive
Strategies" table for a currently-active parameter set (§3b) and no
separate "Feature Weight History"/"Threshold History" tables —
`recommendations WHERE category = ...` (append-only) already is that
history.

**Backtesting Engine (§3c), 8 new tables** — genuinely more new tables
than §3b needed, honestly: `historical_candles` (raw OHLCV cache, unique
on `pair, interval, time`), `backtest_runs` (run config snapshot,
`source_adaptive_strategy_version_id` nullable FK — the link for
backtesting a pending §3b candidate before approving it), `backtest_trades`
(deliberately separate from live `trades` — a backtest re-runs the SAME
historical period many times under different params, which `trades` has
no `run_id` concept for, and conflating them risks simulated data leaking
into live dashboards/risk state), `backtest_portfolio_snapshots`
(mark-to-market equity curve, genuinely new — no existing intraday-equity
drawdown), `backtest_execution_history` (order-lifecycle events —
submitted/filled/partial/rejected/expired, a different grain from
`backtest_trades`' round-trip outcomes), `backtest_performance_metrics`
(`run_id` unique + `metrics jsonb`, same bundle-not-wide-columns pattern
as `strategy_simulations`), `backtest_walk_forward_folds` (real multi-fold
results, deliberately parallel to but separate from `strategy_simulations`
— that one is single-split trade-repartition, this one is genuinely
rolling candle-replay), `backtest_strategy_comparisons` (pairwise run
comparison + p-values + `promotion_recommended`, automatic status marking
only, never automatic deletion/live application). No "Simulation Reports"
table — a report is generated on demand from the above, same precedent as
§3b's report having no storage table of its own.

**Institutional Reliability Layer (§3d), migration `0008`** — 5 new tables
+ 4 new columns, all with safe defaults that preserve today's exact
behavior until a human opts in: `data_quality_log` (one row per issue
found, live or backtest ingestion), `drift_alerts`, `strategy_health_scores`
(a history per version, not just latest-value), `system_metrics` (one
generic table, matching the jsonb-bundle precedent rather than N
single-purpose tables), `circuit_breaker_state` (DB-backed so a trip
survives across cron invocations). New columns:
`capital_config.sizing_mode` (default `'flat'`), `strategy_versions.status`
(default `'active'`), `opportunity_evaluations.config_version` +
`.market_regime`. Unique among this repo's migrations: `get_latest_version()`/
`get_latest_promoted_version()` now filter on `strategy_versions.status`,
so unlike every prior migration (which only left a *new* feature dark
until run), **this one must be applied before its corresponding code
deploys** — those two functions gate every live trading cycle.

**Refinement pass, migration `0009`** — drops one confirmed-exact-duplicate
index on `backtest_portfolio_snapshots` (`0007` had created both a unique
index and a plain index on the identical column tuple); pure cleanup, no
behavior change.

**Scientific Strategy Optimization Framework (§3e), migration `0010`** —
`strategy_versions.promotion_eligible` (default `false`, §2),
`strategy_simulations.research_note`/`.validation_detail`,
`adaptive_strategy_versions.fitness_score` — all nullable/safe-defaulted,
same non-urgent deployment order as every migration except `0008`.

**Promotion Gate (§2/§3e), migration `0011`** — one new table,
`promotion_audit`: `id`, `mode`, `event_type` (`'evaluation'`|
`'promotion'`|`'rollback'`), `candidate_version_id`/`previous_champion_id`/
`new_champion_id` (all `strategy_versions` references), `decision`
(`'PROMOTE'`|`'REJECT'`|`'EXTEND_VALIDATION'`), `promotion_score`,
`gates`/`breakdown`/`reasons` (jsonb — the full evidence bundle
`evaluate_promotion()` built), `created_at`. One row per evaluation, not
just per promotion, so `REJECT`/`EXTEND_VALIDATION` are equally auditable
— same shape/RLS pattern as `drift_alerts`/`strategy_health_scores`
(`0008`). Additive only, no existing-table changes, same non-urgent
deployment order as every migration except `0008`.

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
GitHub Actions (daily cron) ──> evolution_agent.py ────┐  Supabase (Postgres)
                             └─> adaptive_strategy_engine.py ┤◄──────▲
                                 (same job, after evolution_agent)   │
                                                                      │
Vercel (Next.js dashboard) ──── Supabase JS client (anon, RLS) ──────┘
```

The Backtesting Engine (§3c) is deliberately absent from this diagram — it
runs on demand (`python -m src.backtest.engine`/`ingest_data`), never on a
schedule, same precedent as `seed_config.py`. The Institutional Reliability
Layer (§3d) IS in this diagram, but as new steps inside the existing three
jobs, not new nodes: `src.monitoring.diagnostics` joins `risk_check.yml`
(*/5 min), `src.learning.drift_detection`/`strategy_health` join
`evolution.yml` (daily) — no 4th workflow file.

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
9. GitHub Actions workflows for the cron cycle and evolution job
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
│   ├── learning/                    # Trade Memory + Learning Engine (§3a) + Adaptive Strategy Engine (§3b/§3e)
│   │   ├── statistics.py
│   │   ├── trade_memory.py
│   │   ├── confidence_calibration.py
│   │   ├── feature_importance.py
│   │   ├── recommendations.py
│   │   ├── simulation.py            # walk-forward validation, §3b/§3e
│   │   ├── adaptive_strategy_engine.py  # AdaptiveStrategyEngine, §3b/§3e
│   │   ├── fitness.py               # multi-objective fitness score, §3e
│   │   ├── rejection_analysis.py    # root cause analysis, §3e
│   │   ├── weakness_detection.py    # §3e
│   │   ├── evidence_engine.py       # Evidence-Driven Learning Progression, §3e
│   │   ├── learning_status.py       # LearningStatus + can_*() capability gates, §3e
│   │   ├── reports.py
│   │   ├── drift_detection.py       # Feature Drift Detection, §3d
│   │   └── strategy_health.py       # Strategy Health Engine, §3d
│   ├── utils.py                      # shared pure-math helpers (clamp, parse_timestamp, ...)
│   ├── backtest/                    # Event-Driven Backtesting Engine (§3c) — on-demand CLI, not scheduled
│   │   ├── events.py
│   │   ├── simulation_clock.py
│   │   ├── data_provider.py
│   │   ├── order_manager.py
│   │   ├── execution_simulator.py
│   │   ├── portfolio_manager.py
│   │   ├── engine.py                # BacktestEngine, the reactor loop
│   │   ├── performance_analyzer.py
│   │   ├── trade_analysis.py
│   │   ├── walk_forward_validator.py
│   │   ├── strategy_comparison.py
│   │   ├── statistical_validation.py
│   │   ├── overfitting_detection.py
│   │   ├── report.py
│   │   └── ingest_data.py           # CLI: backfills historical_candles
│   ├── data_quality/                # Market Data Quality Engine + Data Repair Engine (§3d)
│   │   ├── validator.py             # MarketDataValidator
│   │   └── repair.py                # DataRepairEngine
│   ├── portfolio/                   # Portfolio Intelligence + Capital Allocation (§3d)
│   │   ├── intelligence.py
│   │   └── capital_allocation.py
│   ├── execution_optimizer/         # Execution Optimizer (§3d)
│   │   └── optimizer.py
│   ├── monitoring/                  # Production Monitoring + Self-Diagnostics (§3d)
│   │   ├── metrics.py
│   │   └── diagnostics.py
│   ├── audit/                       # Audit System (§3d) — read-only decision-trail join
│   │   └── trail.py
│   ├── resilience.py                # retry/backoff + DB-backed circuit breaker (§3d)
│   └── db/
│       ├── models.py
│       └── migrations/
│           ├── 0001_init.sql
│           ├── 0002_rls.sql
│           ├── 0003_pause_flag.sql
│           ├── 0004_opportunity_evaluations.sql
│           ├── 0005_learning_engine.sql
│           ├── 0006_adaptive_strategy_engine.sql
│           ├── 0007_backtesting_engine.sql
│           ├── 0008_reliability_layer.sql
│           ├── 0009_drop_redundant_index.sql
│           └── 0010_scientific_optimization.sql
├── tests/
│   └── test_risk_manager.py
├── dashboard/                      # Next.js app — deferred to step 10
└── .github/workflows/              # deferred to step 9
```

Only directories + `__init__.py` + the real migration SQL are created in
this commit. Agent modules stay unwritten until the plan below is
confirmed — no stub files with fake logic to rewrite later.
