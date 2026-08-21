# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Multi-agent AI crypto trading bot on CoinDCX (INR spot pairs), paper and
real modes sharing one learning/strategy engine — real trading only runs
strategy versions promoted out of paper trading. Full architecture, DB
schema, and build plan: [PROJECT_SPEC.md](PROJECT_SPEC.md) — read it before
any non-trivial change, it is the source of truth, not this file.

Two independent codebases in one repo: a Python trading bot (`src/`, runs
via GitHub Actions cron) and a Next.js dashboard (`dashboard/`, deployed to
Vercel) that reads the same Supabase DB read-only.

## Commands

### Python bot (repo root)

```bash
pip install -r requirements.txt          # deps
pytest                                   # full suite (272 tests, ~1s, all mocked — no network/DB)
pytest tests/test_orchestrator.py        # one file
pytest tests/test_orchestrator.py::test_name  # one test
python3 -m py_compile $(find src -name "*.py")  # syntax-check whole src tree

python3 -m src.seed_config               # bootstrap capital/target config + first strategy version for a mode
python3 -m src.orchestrator --mode=paper # run one trading cycle locally
python3 -m src.agents.evolution_agent    # run nightly learning step locally (feature importance, threshold recs)
python3 -m src.learning.adaptive_strategy_engine  # run adaptive strategy analysis locally
python3 -m src.agents.reporting_agent    # print the HTML report to stdout
```

No lint config in the Python tree (no ruff/flake8/black configured) — match existing style, don't introduce a formatter unasked.

### Dashboard (`dashboard/`)

```bash
npm run dev     # localhost:3000, needs dashboard/.env.local (Supabase URL + anon key)
npm run build
npm run lint    # eslint
```

## Architecture

### Pipeline (`src/orchestrator.py::run_cycle`)

Quant-first, LLM-gated, two-pass per cycle:

1. **Data Agent** (`src/agents/data_agent.py`) — pulls candles from CoinDCX for `FEATURE_TIMEFRAMES`.
2. **Feature Engine** (`src/features/feature_engine.py`) — pure indicator math (RSI, MACD, StochRSI, ATR, Bollinger, OBV, ADX, EMAs, support/resistance), no LLM.
3. **Opportunity Scorer** (`src/features/opportunity_scorer.py`) — deterministic weighted blend of 5 sub-scores (trend/momentum/volume/volatility/risk, `OPPORTUNITY_WEIGHT_*` in config) into one 0-100 score per symbol; also classifies market regime. `weighted_average()` is the public entry point other modules reuse. Only the `TOP_N_CANDIDATES` above `MIN_OPPORTUNITY_SCORE` go to the LLM at all — this is the cost/rate-limit control.
4. **Signal Agent** (`src/agents/signal_agent.py`) — LLM (Groq default, Ollama Cloud / Gemini alternatives via `LLM_PROVIDER`, model-chain fallback) validates/rejects the quant candidates and returns structured reasoning; never invents candidates itself.
5. **Risk Manager** (`src/agents/risk_manager.py`) — position sizing, stop-loss/take-profit, circuit breaker, daily loss limits. Pure Python, no LLM.
6. **Execution Agent** (`src/agents/execution/`) — `ExecutionAgent` ABC, `PaperExecutionAgent` (simulated fills) and `RealExecutionAgent` (live CoinDCX orders — wired in but its order-placement path is unverified against a live fill; stays inert until a strategy version clears the promotion bar in PROJECT_SPEC.md §2).

Pass 1 of `run_cycle` handles open-position exits (stop-loss/take-profit/exit-score); pass 2 evaluates new entries against scored candidates. `.github/workflows/risk_check.yml` runs this exit sweep every 5 minutes independent of the full 10-minute signal cycle (`trading_cycle.yml`) — polling-based, not an exchange-side stop order, so it's not a hard guarantee.

### Learning system (`src/learning/`) — two trust tiers

**Trade Memory + Learning Engine** (§3a in PROJECT_SPEC.md): `trade_memory.py` (nearest-neighbor similar-trade lookup for confidence blending), `feature_importance.py` (per-timeframe + blended sub-score correlation), `statistics.py` (Sharpe/Sortino/Calmar/expectancy/streaks/z-tests, all stdlib — no numpy/scipy anywhere in this repo), `confidence_calibration.py`, `recommendations.py`, `reports.py`. Runs nightly via `evolution_agent.py`.

**Adaptive Strategy Intelligence Engine** (§3b, `adaptive_strategy_engine.py` + `simulation.py`) — layered on top, runs as its **own** independent nightly step (`python -m src.learning.adaptive_strategy_engine` in `evolution.yml`, after but not inside `evolution_agent.run_evolution()` — never merge these two call sites, it was deliberately kept as a separate step to avoid coupling). Generates weight/regime/symbol/threshold recommendations, walk-forward validates them (train/test time-split, no look-ahead), simulates against trades actually taken (never fabricates counterfactual trades — no backtester exists), and versions passing candidates into `adaptive_strategy_versions`.

**Invariant, narrowed**: nothing in `src/learning/` ever auto-writes to `config.py` or live scoring weights (`OPPORTUNITY_WEIGHT_*` and similar `os.getenv()` constants) — weight/regime/symbol-threshold recommendations stay advisory, a human still reviews `recommendations`/`adaptive_strategy_versions` rows in Supabase's table editor and manually copies values into env vars, exactly as before. What's no longer true as a blanket statement: **exit-params candidates** (`stop_loss_pct`/`take_profit_pct`, which target a DB row — `strategy_versions.params_json` — not an env var) **auto-activate** into a new `strategy_versions` row the moment they clear every statistical gate (`simulation.py::_activate_exit_params_candidate`), and `evolution_agent.py::run_evolution()` auto-flips `promoted_to_real` the same way once `promotion_eligible()`'s five gates (§2) all pass — see PROJECT_SPEC.md §2/§3b for the full reasoning (unchanged gates, only the human click removed; a fresh version's own `PROMOTION_MIN_PAPER_DAYS` clock still starts at zero). The confidence-modifier chain in `confidence_calibration.py` (regime/symbol/recent-performance modifiers) remains the third automatic piece, extending the already-automatic, already-inert-by-default gate (`MIN_FINAL_CONFIDENCE=0`). Don't treat any of these three as precedent for auto-applying weight/regime/symbol-threshold recommendations too — that still needs env-var rewrites + a redeploy, a deliberately separate, bigger change.

### Config (`src/config.py`)

Every tunable is `os.getenv(...) or default` — no hardcoded constants scattered in logic. When adding a new tunable, add it here following that exact pattern, then to `.env.example` with a comment, matching the existing grouped-by-feature layout.

### Database (Supabase/Postgres, `src/db/`)

`models.py` is the only module that talks to Supabase — every DB access in `src/` goes through it, never a raw client call elsewhere. Migrations are numbered SQL files in `src/db/migrations/`, applied manually in Supabase's SQL Editor (no migration runner) — after adding one, tell the user it needs a manual run, don't assume it's live. Every table follows the same RLS pattern: `enable row level security`, public-read select policy, writes via service key only. This codebase has an explicit precedent against storing one fact under two names (e.g. no separate `recovery_factor` column alongside `calmar_ratio`, no separate `market_regimes` table when `learning_statistics WHERE dimension_type='market_regime'` already answers it) — check for an existing table/column before adding a new one for a derivable fact.

### Dashboard (`dashboard/src/`)

Next.js App Router, client components fetch Supabase directly (anon key, RLS-gated read-only) — no API routes/server actions proxying it. Pages: `/` (overview stat cards), `/trades`, `/evolution` (PnL chart + version history), `/model-health`, `/config` (gated behind Supabase Auth sign-in, no auth user exists yet so it's expected to stay on the login form). Paper/real mode selected via `?mode=` query param, shared via `ModeToggle.tsx`.

### Testing conventions

All 272 tests mock `src.db.models` / external clients — no real network or DB calls in the suite (a slow/hanging test run is a signal something is leaking a real client, not just flakiness). New statistical or DB-touching code gets its own `tests/test_<module>.py`; don't rely on indirect coverage through a caller's mocked test.

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy to keep main context window clean

- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -> then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to 'tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review to 'tasks/todo.md'
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
