# Deploying to Google Cloud Run Jobs + Cloud Scheduler

**Not actually deployed — confirmed directly with the repo owner (no
gcloud CLI set up, never run). This describes a planned alternative to
the GitHub Actions crons, not what's live.** The GH Actions workflows
(`.github/workflows/*.yml`) are the real, live execution path — an
external pinger hits `dashboard/src/app/api/cron/[workflow]/route.ts`
on a real clock, which dispatches `trading_cycle.yml`/`risk_check.yml`
via the GitHub API (`evolution.yml` has its own native hourly
`schedule:` trigger, reliable enough at that cadence). Treat everything
below as a design doc for a future migration, not ground truth about
what's running right now.

---

Replaces GitHub Actions as the execution layer (private repo was blowing
past the free 2,000 min/month). Code stays private; Cloud Run Jobs +
Cloud Scheduler are free at this scale (Job execution well within the
free tier's vCPU-second/request allowance; Scheduler's free tier is
capped at 3 jobs/month forever — this setup uses exactly 3).

Replace `PROJECT_ID` below with your real GCP project ID throughout.
Region used: `asia-south1` (closest to CoinDCX/India) — change every
`asia-south1` below consistently if you'd rather use a different one.

## One-time setup

```bash
gcloud config set project PROJECT_ID

gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com
```

### 1. Secrets

```bash
echo -n "VALUE" | gcloud secrets create LLM_PROVIDER --data-file=-
echo -n "VALUE" | gcloud secrets create GROQ_API_KEY --data-file=-
echo -n "VALUE" | gcloud secrets create GROQ_MODEL_CHAIN --data-file=-
echo -n "VALUE" | gcloud secrets create OLLAMA_API_KEY --data-file=-
echo -n "VALUE" | gcloud secrets create OLLAMA_BASE_URL --data-file=-
echo -n "VALUE" | gcloud secrets create OLLAMA_MODEL_CHAIN --data-file=-
echo -n "VALUE" | gcloud secrets create DATABASE_URL --data-file=-
echo -n "VALUE" | gcloud secrets create COINDCX_API_KEY --data-file=-
echo -n "VALUE" | gcloud secrets create COINDCX_API_SECRET --data-file=-
gcloud secrets create FIREBASE_SERVICE_ACCOUNT_JSON --data-file=service-account.json
```

(Migrating off Supabase: `DATABASE_URL` is Neon's **pooled** connection
string — the same one `src/db/models.py` reads locally via `.env`. If
`SUPABASE_URL`/`SUPABASE_SERVICE_KEY` secrets already exist from before,
delete them once every job below has been redeployed and verified:
`gcloud secrets delete SUPABASE_URL SUPABASE_SERVICE_KEY`.)

(Migrating off Neon onto Firebase/Firestore, table-by-table — see
`src/db/models.py`'s module docstring: `FIREBASE_SERVICE_ACCOUNT_JSON`
is required starting Phase 1 alongside `DATABASE_URL`, not replacing it
— every table not yet on its own migration phase still needs Neon.
`service-account.json` is the key file downloaded from Firebase Console
→ Project Settings → Service Accounts → Generate new private key,
project `ai-bot-14723`; `--data-file=` uploads its contents as the
secret value, same idea as the inline `echo -n` calls above but for a
whole file. `DATABASE_URL` can be deleted once every table has migrated
— not yet.)

(Updating a value later: `echo -n "NEW_VALUE" | gcloud secrets versions add NAME --data-file=-`,
or `gcloud secrets versions add FIREBASE_SERVICE_ACCOUNT_JSON --data-file=service-account.json`
for the key file.)

### 2. Deploy the 3 Cloud Run Jobs

`--source .` builds and pushes the image via Cloud Build automatically —
no manual `docker build`/`docker push` needed. Run from the repo root.

**`--max-retries=0` is deliberate, not an oversight** — Cloud Run Jobs
defaults to 3 retries on failure. A trading cycle that failed partway
(e.g. after placing an order but before recording it) must NOT be
silently re-run — that's how a bot double-trades. GH Actions never
retried these either; this preserves that.

```bash
gcloud run jobs deploy trading-cycle \
  --source . \
  --region=asia-south1 \
  --command=bash \
  --args=deploy/run_trading_cycle.sh \
  --max-retries=0 \
  --set-secrets=LLM_PROVIDER=LLM_PROVIDER:latest,GROQ_API_KEY=GROQ_API_KEY:latest,GROQ_MODEL_CHAIN=GROQ_MODEL_CHAIN:latest,OLLAMA_API_KEY=OLLAMA_API_KEY:latest,OLLAMA_BASE_URL=OLLAMA_BASE_URL:latest,OLLAMA_MODEL_CHAIN=OLLAMA_MODEL_CHAIN:latest,DATABASE_URL=DATABASE_URL:latest,COINDCX_API_KEY=COINDCX_API_KEY:latest,COINDCX_API_SECRET=COINDCX_API_SECRET:latest,FIREBASE_SERVICE_ACCOUNT_JSON=FIREBASE_SERVICE_ACCOUNT_JSON:latest

gcloud run jobs deploy risk-check \
  --source . \
  --region=asia-south1 \
  --command=bash \
  --args=deploy/run_risk_check.sh \
  --max-retries=0 \
  --set-secrets=DATABASE_URL=DATABASE_URL:latest,COINDCX_API_KEY=COINDCX_API_KEY:latest,COINDCX_API_SECRET=COINDCX_API_SECRET:latest,FIREBASE_SERVICE_ACCOUNT_JSON=FIREBASE_SERVICE_ACCOUNT_JSON:latest

gcloud run jobs deploy evolution \
  --source . \
  --region=asia-south1 \
  --command=bash \
  --args=deploy/run_evolution.sh \
  --max-retries=0 \
  --set-secrets=DATABASE_URL=DATABASE_URL:latest,FIREBASE_SERVICE_ACCOUNT_JSON=FIREBASE_SERVICE_ACCOUNT_JSON:latest
```

### 3. Invoker service account (lets Cloud Scheduler trigger the Jobs)

```bash
gcloud iam service-accounts create scheduler-invoker \
  --display-name="Cloud Scheduler -> Cloud Run Jobs invoker"

for JOB in trading-cycle risk-check evolution; do
  gcloud run jobs add-iam-policy-binding "$JOB" \
    --region=asia-south1 \
    --member="serviceAccount:scheduler-invoker@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker"
done
```

### 4. Cloud Scheduler — the 3 triggers

Risk-check runs every minute, not every 5 — a position whose stop/target
was hit could otherwise sit unwatched for most of a 5min gap and turn a
winner into a loss before the next poll (`src/orchestrator.py::
run_risk_check` guards this cadence with a DB mutex — an overlapping run
skips instead of double-closing a position — see migration
`0015_risk_check_lock.sql`). Trading-cycle stays at 10min (throttled by
feature-computation cost, not by this concern).

**Already deployed?** Update the existing job in place instead of
`create` (which errors on a name that already exists):
```bash
gcloud scheduler jobs update http risk-check-trigger \
  --location=asia-south1 \
  --schedule="* * * * *"
```
Then apply migration `0015_risk_check_lock.sql` against Neon before the
next trigger fires (the app doesn't need to be redeployed for a Job —
just re-run `gcloud run jobs deploy risk-check --source . ...` from
step 2 once, so the running image has the mutex code).

```bash
gcloud scheduler jobs create http trading-cycle-trigger \
  --location=asia-south1 \
  --schedule="2-59/10 * * * *" \
  --uri="https://asia-south1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/PROJECT_ID/jobs/trading-cycle:run" \
  --http-method=POST \
  --oauth-service-account-email="scheduler-invoker@PROJECT_ID.iam.gserviceaccount.com"

gcloud scheduler jobs create http risk-check-trigger \
  --location=asia-south1 \
  --schedule="* * * * *" \
  --uri="https://asia-south1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/PROJECT_ID/jobs/risk-check:run" \
  --http-method=POST \
  --oauth-service-account-email="scheduler-invoker@PROJECT_ID.iam.gserviceaccount.com"

# 18:45 UTC = 00:15 IST, just after the trading-day rollover — same as evolution.yml
gcloud scheduler jobs create http evolution-trigger \
  --location=asia-south1 \
  --schedule="45 18 * * *" \
  --uri="https://asia-south1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/PROJECT_ID/jobs/evolution:run" \
  --http-method=POST \
  --oauth-service-account-email="scheduler-invoker@PROJECT_ID.iam.gserviceaccount.com"
```

## Verification

1. **Local, before any of the above** — catches a packaging mistake cheaply:
   ```bash
   docker build -t ai-trader .
   docker run --rm \
     -e DATABASE_URL=... \
     -e LLM_PROVIDER=... -e GROQ_API_KEY=... \
     ai-trader bash deploy/run_trading_cycle.sh
   ```
   (Uses the real paper-mode Neon database — safe, same DB the bot
   already writes to every cycle.)

2. **After deploying each Job**, fire it manually once and check both
   the execution log AND that a real row landed in Neon — a clean
   exit code alone doesn't prove the trading logic ran:
   ```bash
   gcloud run jobs execute trading-cycle --region=asia-south1
   gcloud run jobs execute risk-check --region=asia-south1
   gcloud run jobs execute evolution --region=asia-south1
   ```
   Check `opportunity_evaluations`/`trades`/`model_usage` (trading-cycle,
   risk-check) or `recommendations`/`adaptive_strategy_versions`
   (evolution) for a fresh timestamp.

3. **After creating the Scheduler jobs**, watch Cloud Run's execution
   history for the next 30-60 min to confirm cadence actually holds —
   same discipline as the earlier GH Actions cron fix, since drift is
   exactly the failure mode this migration exists to fix.

## Not done here, on purpose

The 3 `.github/workflows/*.yml` files and the Vercel `/api/cron/[workflow]`
trigger route are left in place, untouched — they're already
`workflow_dispatch`-only so nothing auto-fires from them, and keeping
them gives a manual fallback during cutover. Once Cloud Scheduler has
been running cleanly for a while, both can be deleted as a follow-up
(the Vercel route's `CRON_TRIGGER_SECRET`/`GITHUB_DISPATCH_PAT` env vars
and the two cron-job.org jobs go with it).
