# AI-Trader

Multi-agent AI crypto trading bot on CoinDCX (INR pairs). Paper and real
trading modes share one learning/strategy engine; real trading only runs
strategy versions promoted out of paper trading.

Full architecture, database schema, and build plan: [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Status

All 11 build-order steps done. `RealExecutionAgent` (step 11) is wired
in but stays inert until a strategy version actually clears the
promotion bar (see PROJECT_SPEC.md §2) — and its order-placement path
is unverified against a live fill (the account had ~₹0.91 balance at
build time, under CoinDCX's ₹100 min_notional). Confirm with one small
real order once funds exist, before trusting it for real money.

## Setup

1. **Install deps**: `pip install -r requirements.txt`
2. **Supabase**: create a free-tier project, then run
   [`src/db/migrations/0001_init.sql`](src/db/migrations/0001_init.sql) in
   its SQL Editor.
3. **Env vars**: copy `.env.example` to `.env` and fill in `GROQ_API_KEY`,
   `COINDCX_API_KEY`/`COINDCX_API_SECRET` (only needed once step 11's real
   execution lands), and the Supabase URL/service key from your project
   settings.
4. **Seed config**: `python3 -m src.seed_config` — sets capital/target/
   loss for a mode and bootstraps the first strategy version. Throwaway,
   replaced by the dashboard's Config panel at step 10.
5. **Run a cycle locally**: `python3 -m src.orchestrator --mode=paper`
6. **Run the tests**: `pytest`

## GitHub Actions

Two scheduled workflows call the same code as local runs — set these as
repo secrets (Settings → Secrets and variables → Actions) so they can:

- `GROQ_API_KEY`, `GROQ_MODEL_CHAIN`
- `COINDCX_API_KEY`, `COINDCX_API_SECRET`
- `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`

`.github/workflows/trading_cycle.yml` runs every 10 minutes (paper and
real, in parallel — real no-ops until a strategy is promoted).
`.github/workflows/evolution.yml` runs nightly, just after the IST
trading-day rollover.
