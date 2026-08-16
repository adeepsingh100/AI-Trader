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
`src/agents/`. One full cycle (data → signal → risk → execute → log)
runs to completion and exits — no long-running process (see §7).

### Data Agent (`src/agents/data_agent.py`)
- Pulls CoinDCX's public market endpoints (ticker, orderbook, candles).
- Each cycle: fetches 24h ticker volume for all INR pairs, selects the
  **top 10 by volume**, pulls candles/orderbook for those 10. (Resolved
  decision — see §9: dynamic top-10-by-volume, not a fixed pair list.)
- No auth required (public endpoints only).

### Signal Agent (`src/agents/signal_agent.py`)
- For each of the top-10 candidates, calls the LLM (via Groq, see §4)
  with market data + the active strategy version's prompt/params.
- Returns a signal per symbol: direction, confidence, reasoning text.
  Reasoning text is always persisted (`trades.reasoning_text`).

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
- From the top-10 signals, only takes symbols with an open sizing slot
  and a non-flat signal; ties broken by signal confidence.
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

### Reporting Agent (`src/agents/reporting_agent.py`)
- Generates an HTML report covering both modes side by side: PnL vs
  target, trade log, current strategy version + changelog, model
  fallback stats (from `model_usage`).

### Orchestrator (`src/orchestrator.py`)
- Single script invocation, one full cycle, then exit. Invoked per mode
  (paper, real) — real invocation is a no-op if no version is promoted.
- Sequence: check circuit-breaker state for today → Data Agent → Signal
  Agent → Risk Manager → Execution Agent → persist to `trades` /
  `daily_pnl` / `agent_logs`.

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
  id             bigserial primary key,
  mode           text not null,              -- 'paper' | 'real'
  version_id     bigint not null references strategy_versions(id),
  symbol         text not null,              -- e.g. 'BTCINR'
  side           text not null,              -- 'buy' | 'sell'
  qty            numeric not null,
  entry_price    numeric not null,
  exit_price     numeric,
  pnl            numeric,
  fees           numeric not null default 0,
  status         text not null,              -- 'open' | 'closed' | 'flattened'
  opened_at      timestamptz not null default now(),
  closed_at      timestamptz,
  reasoning_text text                        -- full LLM reasoning for this trade
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
```

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
│   │   ├── signal_agent.py
│   │   ├── risk_manager.py
│   │   ├── evolution_agent.py
│   │   ├── reporting_agent.py
│   │   └── execution/
│   │       ├── base.py
│   │       ├── paper.py
│   │       └── real.py
│   └── db/
│       ├── models.py
│       └── migrations/
│           └── 0001_init.sql
├── tests/
│   └── test_risk_manager.py
├── dashboard/                      # Next.js app — deferred to step 10
└── .github/workflows/              # deferred to step 9
```

Only directories + `__init__.py` + the real migration SQL are created in
this commit. Agent modules stay unwritten until the plan below is
confirmed — no stub files with fake logic to rewrite later.
