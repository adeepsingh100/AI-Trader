# Running the bot 24/7 on Windows (replaces GCP Cloud Run Jobs + Scheduler)

Same 3 jobs as `deploy/README.md`'s GCP setup, same cadence, running as
native Windows Scheduled Tasks against the same Neon database instead:

| Task | Runs | Interval |
|---|---|---|
| `AI-Trader-TradingCycle` | `src.orchestrator --mode=paper/real` | 10 min |
| `AI-Trader-RiskCheck` | `src.orchestrator --risk-only` + diagnostics | 1 min |
| `AI-Trader-Evolution` | evolution/adaptive/drift/health chain | 60 min |

## One-time setup

```powershell
# 1. Python deps, in a dedicated venv (the scheduled tasks call this
#    venv's python.exe directly, not whatever's on PATH)
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. .env at the repo root — same DATABASE_URL/COINDCX_*/GROQ_*/GEMINI_*
#    values you already use for local dev. If you don't have one yet,
#    copy .env.example and fill in real values.

# 3. Register the 3 scheduled tasks — run as Administrator
powershell -ExecutionPolicy Bypass -File deploy\windows\register_tasks.ps1
```

`register_tasks.ps1` prompts once for your Windows account password —
Task Scheduler stores it (not this script) so tasks keep running after
you log out or lock the screen. Tasks are set to survive reboots, run on
battery, and auto-restart up to 3 times if one crashes.

## Verify

```powershell
Get-ScheduledTask -TaskName "AI-Trader-*" | Get-ScheduledTaskInfo
```

Check `LastTaskResult` is `0` after each has fired once (risk-check
within a minute, trading-cycle within 10). Tail the logs:

```powershell
Get-Content logs\risk_check.log -Wait -Tail 20
```

Then confirm real rows are landing — same discipline as the GCP
verification steps: query `opportunity_evaluations`/`trades` for a fresh
timestamp, not just a clean exit code.

## Critical: don't run this alongside GCP

Both point at the same Neon database. `run_risk_check` has a DB mutex
(`risk_check_lock`, migration `0015`) that protects against two overlapping
risk-check runs double-closing a position — that mutex is keyed in the
shared DB, so it also protects across GCP-vs-Windows overlap. **Trading-cycle
has no equivalent guard**: two sources both opening entries in the same
window can double-count available capital and open duplicate positions.

Before or immediately after running `register_tasks.ps1`, pause GCP's
triggers so only one source is ever live:

```bash
gcloud scheduler jobs pause trading-cycle-trigger --location=asia-south1
gcloud scheduler jobs pause risk-check-trigger --location=asia-south1
gcloud scheduler jobs pause evolution-trigger --location=asia-south1
```

(`pause`, not `delete` — reversible if the Windows machine ever goes
down and you need GCP back: `gcloud scheduler jobs resume <name>
--location=asia-south1`.)

## Stop / undo

```powershell
powershell -ExecutionPolicy Bypass -File deploy\windows\unregister_tasks.ps1
```
