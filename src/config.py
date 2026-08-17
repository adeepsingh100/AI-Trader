from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider (Groq / Ollama Cloud) --------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_CHAIN = [
    m.strip()
    # `or` (not getenv's default arg) — GitHub Actions sets a referenced
    # secret that was never created as an empty string, not an absent
    # key, so getenv's own default never kicks in.
    for m in (os.getenv("GROQ_MODEL_CHAIN") or "openai/gpt-oss-120b,qwen/qwen3.6-27b").split(",")
    if m.strip()
]

# "groq" (default) or "ollama" (Ollama Cloud — https://ollama.com, not a
# local instance). Switch by setting LLM_PROVIDER, no code change needed.
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "https://ollama.com"
OLLAMA_MODEL_CHAIN = [
    m.strip() for m in (os.getenv("OLLAMA_MODEL_CHAIN") or "gpt-oss:120b").split(",") if m.strip()
]

# --- Credentials -------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "")
COINDCX_API_SECRET = os.getenv("COINDCX_API_SECRET", "")

# --- Promotion (paper -> real) -----------------------------------------------
PROMOTION_MIN_PAPER_DAYS = int(os.getenv("PROMOTION_MIN_PAPER_DAYS", "14"))
PROMOTION_MIN_CUMULATIVE_PNL = float(os.getenv("PROMOTION_MIN_CUMULATIVE_PNL", "0"))
PROMOTION_MAX_DRAWDOWN_PCT = float(os.getenv("PROMOTION_MAX_DRAWDOWN_PCT", "15"))

# --- Feature Engine / Opportunity Scorer -----------------------------------
# Quant-first pipeline: Feature Engine computes indicators per timeframe,
# Opportunity Scorer turns them into a deterministic 0-100 score, only the
# top candidates go to the LLM for accept/reject validation. Every knob
# below is named and overridable so nothing in that path is a bare literal.

# tf:weight pairs, e.g. "1m:0.15,15m:0.25,1h:0.30,1d:0.30" — weights used to
# blend a per-timeframe score into one sub-score. Must sum to ~1.0 (scorer
# renormalizes defensively regardless). FEATURE_TIMEFRAMES (which candles to
# fetch) is derived from this dict's keys rather than a second env var, so
# "which timeframes get fetched" and "which get weighted" can't drift apart.
# CoinDCX's public candles API only accepts interval in {1m, 15m, 1h, 1d} —
# any other value (e.g. 5m, 4h) 422s, so keys here must stay within that set.
TIMEFRAME_WEIGHTS = {
    tf: float(weight)
    for tf, weight in (
        pair.split(":")
        for pair in (os.getenv("TIMEFRAME_WEIGHTS") or "1m:0.15,15m:0.25,1h:0.30,1d:0.30").split(",")
        if pair.strip()
    )
}
FEATURE_TIMEFRAMES = list(TIMEFRAME_WEIGHTS.keys())

FEATURE_CANDLE_LIMIT = int(os.getenv("FEATURE_CANDLE_LIMIT", "250"))

# Indicator periods (Wilder-smoothed where that's the standard convention).
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
MACD_FAST = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", "9"))
STOCH_RSI_PERIOD = int(os.getenv("STOCH_RSI_PERIOD", "14"))
STOCH_K_SMOOTH = int(os.getenv("STOCH_K_SMOOTH", "3"))
STOCH_D_SMOOTH = int(os.getenv("STOCH_D_SMOOTH", "3"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
BOLLINGER_PERIOD = int(os.getenv("BOLLINGER_PERIOD", "20"))
BOLLINGER_NUM_STD = float(os.getenv("BOLLINGER_NUM_STD", "2"))
RELATIVE_VOLUME_LOOKBACK = int(os.getenv("RELATIVE_VOLUME_LOOKBACK", "20"))
VOLUME_SPIKE_THRESHOLD = float(os.getenv("VOLUME_SPIKE_THRESHOLD", "2.0"))
OBV_SLOPE_LOOKBACK = int(os.getenv("OBV_SLOPE_LOOKBACK", "5"))
SUPPORT_RESISTANCE_LOOKBACK = int(os.getenv("SUPPORT_RESISTANCE_LOOKBACK", "50"))
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
EMA_TREND_PERIOD_1 = int(os.getenv("EMA_TREND_PERIOD_1", "20"))
EMA_TREND_PERIOD_2 = int(os.getenv("EMA_TREND_PERIOD_2", "50"))
EMA_TREND_PERIOD_3 = int(os.getenv("EMA_TREND_PERIOD_3", "100"))
EMA_TREND_PERIOD_4 = int(os.getenv("EMA_TREND_PERIOD_4", "200"))

# Sub-score mapping thresholds. RSI and StochRSI get separate floor/ceil
# pairs deliberately — StochRSI hugs 0/100 far harder than RSI, so reusing
# RSI's 30/70 band against StochRSI would misgrade momentum.
RSI_SCORE_FLOOR = float(os.getenv("RSI_SCORE_FLOOR", "30"))
RSI_SCORE_CEIL = float(os.getenv("RSI_SCORE_CEIL", "70"))
STOCH_RSI_SCORE_FLOOR = float(os.getenv("STOCH_RSI_SCORE_FLOOR", "20"))
STOCH_RSI_SCORE_CEIL = float(os.getenv("STOCH_RSI_SCORE_CEIL", "80"))
VOLUME_SCORE_SCALE = float(os.getenv("VOLUME_SCORE_SCALE", "50"))
VOLATILITY_LOW_MAX_PCT = float(os.getenv("VOLATILITY_LOW_MAX_PCT", "0.5"))
VOLATILITY_HIGH_MIN_PCT = float(os.getenv("VOLATILITY_HIGH_MIN_PCT", "5.0"))
if VOLATILITY_LOW_MAX_PCT >= VOLATILITY_HIGH_MIN_PCT:
    raise ValueError(
        "VOLATILITY_LOW_MAX_PCT must be < VOLATILITY_HIGH_MIN_PCT "
        f"(got {VOLATILITY_LOW_MAX_PCT} >= {VOLATILITY_HIGH_MIN_PCT})"
    )
VOLATILITY_SCORE_EXTREME = float(os.getenv("VOLATILITY_SCORE_EXTREME", "40"))
RISK_RESISTANCE_DISTANCE_FOR_MAX_SCORE = float(
    os.getenv("RISK_RESISTANCE_DISTANCE_FOR_MAX_SCORE", "5.0")
)

# Final opportunity_score = weighted blend of the 5 sub-scores. Scorer
# renormalizes these to sum to 1.0 at call time, so a misconfigured sum
# doesn't silently distort the 0-100 scale.
OPPORTUNITY_WEIGHT_TREND = float(os.getenv("OPPORTUNITY_WEIGHT_TREND", "0.30"))
OPPORTUNITY_WEIGHT_MOMENTUM = float(os.getenv("OPPORTUNITY_WEIGHT_MOMENTUM", "0.25"))
OPPORTUNITY_WEIGHT_VOLUME = float(os.getenv("OPPORTUNITY_WEIGHT_VOLUME", "0.15"))
OPPORTUNITY_WEIGHT_VOLATILITY = float(os.getenv("OPPORTUNITY_WEIGHT_VOLATILITY", "0.15"))
OPPORTUNITY_WEIGHT_RISK = float(os.getenv("OPPORTUNITY_WEIGHT_RISK", "0.15"))

# Candidate filtering: only the top N not-held symbols scoring >= the
# minimum go to LLM validation for entry. A held symbol whose recomputed
# score falls below EXIT_SCORE_THRESHOLD becomes an LLM exit-validation
# candidate. EXIT_SCORE_THRESHOLD must stay < MIN_OPPORTUNITY_SCORE — that
# gap is a hysteresis band so a symbol scoring in between isn't
# simultaneously too-weak-to-enter and forced-to-exit-if-held.
TOP_N_CANDIDATES = int(os.getenv("TOP_N_CANDIDATES", "5"))
MIN_OPPORTUNITY_SCORE = float(os.getenv("MIN_OPPORTUNITY_SCORE", "60"))
EXIT_SCORE_THRESHOLD = float(os.getenv("EXIT_SCORE_THRESHOLD", "40"))
if EXIT_SCORE_THRESHOLD >= MIN_OPPORTUNITY_SCORE:
    raise ValueError(
        "EXIT_SCORE_THRESHOLD must be < MIN_OPPORTUNITY_SCORE "
        f"(got {EXIT_SCORE_THRESHOLD} >= {MIN_OPPORTUNITY_SCORE})"
    )

# --- Market regime classification -------------------------------------------
# Folded into score_opportunity()'s output (src/features/opportunity_scorer.py)
# — reuses trend_score/ADX/volatility_regime, no separate computation pass.
REGIME_ADX_TREND_THRESHOLD = float(os.getenv("REGIME_ADX_TREND_THRESHOLD", "20"))
REGIME_STRONG_TREND_SCORE_MIN = float(os.getenv("REGIME_STRONG_TREND_SCORE_MIN", "75"))

# --- Trade Memory / Learning Engine ------------------------------------------
# Every completed trade is stored with full entry-time context, statistically
# analyzed per bucket, and searched for similarity before the next entry —
# see PROJECT_SPEC.md §3a. Pure statistics, no ML/RL. Outputs here are only
# as reliable as trade volume allows; the zero-denominator guards throughout
# src/learning/ exist so thin history degrades to "unknown", never a faked
# number.

# Shared by similarity search AND learning_statistics bucket recompute — one
# dial, not two, so both look at the same "current" window as trade history
# grows and old strategy-version-era trades age out.
LEARNING_HISTORY_WINDOW_DAYS = int(os.getenv("LEARNING_HISTORY_WINDOW_DAYS", "180"))
# How far back process_closed_trades() looks for trades it hasn't evaluated
# yet — generous margin so a missed cron run still gets caught up.
LEARNING_CATCHUP_LOOKBACK_HOURS = int(os.getenv("LEARNING_CATCHUP_LOOKBACK_HOURS", "72"))

MAX_SIMILAR_TRADES_SCANNED = int(os.getenv("MAX_SIMILAR_TRADES_SCANNED", "500"))
MIN_SIMILAR_TRADES = int(os.getenv("MIN_SIMILAR_TRADES", "5"))
SIMILARITY_TOP_N = int(os.getenv("SIMILARITY_TOP_N", "10"))
# Euclidean distance over 5 sub-scores each 0-100 -> max possible ~224
# (100*sqrt(5)). 30 ~= 13.4 pts average per-dimension divergence, a fairly
# tight band given the sub-scores are themselves coarse weighted averages.
SIMILARITY_MAX_DISTANCE = float(os.getenv("SIMILARITY_MAX_DISTANCE", "30"))

# Minimum acceptable return for Sortino's downside deviation (0 = any loss
# counts as downside).
SORTINO_MAR_PCT = float(os.getenv("SORTINO_MAR_PCT", "0"))

# Final confidence = ai_confidence*AI_WEIGHT + historical_confidence*HISTORICAL_WEIGHT,
# renormalized defensively like OPPORTUNITY_WEIGHT_*. Collapses to AI-only
# when history is too thin (see MIN_SIMILAR_TRADES).
CONFIDENCE_AI_WEIGHT = float(os.getenv("CONFIDENCE_AI_WEIGHT", "0.6"))
CONFIDENCE_HISTORICAL_WEIGHT = float(os.getenv("CONFIDENCE_HISTORICAL_WEIGHT", "0.4"))
# Hard gate before place_order, alongside the existing risk_manager check.
# Default 0 = permissive/inert until deliberately tightened once enough
# trade history exists to trust calibrated confidence.
MIN_FINAL_CONFIDENCE = float(os.getenv("MIN_FINAL_CONFIDENCE", "0"))

# Self-evaluation (trade_evaluations): was the predicted confidence/score
# actually right, checked against these midpoints after the fact.
CONFIDENCE_ACCURACY_MIDPOINT = float(os.getenv("CONFIDENCE_ACCURACY_MIDPOINT", "50"))

# learning_statistics bucket widths for the opportunity_score/confidence
# dimensions (e.g. width 10 -> buckets "60-70", "70-80", ...).
OPPORTUNITY_SCORE_BUCKET_WIDTH = float(os.getenv("OPPORTUNITY_SCORE_BUCKET_WIDTH", "10"))
CONFIDENCE_BUCKET_WIDTH = float(os.getenv("CONFIDENCE_BUCKET_WIDTH", "10"))

# recommendations (Step 8, advisory only, never auto-applied) and
# feature_importance both need enough samples before a suggestion/correlation
# means anything rather than reading noise as signal — same floor, one knob.
# Also reused (deliberately, not duplicated) as the "trust this bucket"
# floor for the adaptive confidence modifiers below.
RECOMMENDATION_MIN_IMPROVEMENT_PCT = float(os.getenv("RECOMMENDATION_MIN_IMPROVEMENT_PCT", "15"))
RECOMMENDATION_MIN_SAMPLE_SIZE = int(os.getenv("RECOMMENDATION_MIN_SAMPLE_SIZE", "20"))
# Absolute expectancy-delta floor, alongside the relative-percent check
# above — needed once recommendations get generated per symbol/regime
# bucket, where a near-zero baseline expectancy makes the relative-percent
# check trivially pass on a meaningless swing.
MIN_EXPECTANCY_DELTA = float(os.getenv("MIN_EXPECTANCY_DELTA", "1.0"))

# --- Adaptive Strategy Intelligence Engine -----------------------------------
# Closes the loop from the Learning Engine's statistics back into advisory
# recommendations — never auto-applied to config/live scoring (human
# approves in Supabase, same as `recommendations` already works). The one
# automatic piece is the confidence-modifier chain below, which extends the
# already-automatic (and inert-by-default, MIN_FINAL_CONFIDENCE=0)
# calibrate_confidence gate. Pure statistics throughout — no ML/RL.

# Walk-forward validation: generate a recommendation using only the older
# TRAIN fraction of LEARNING_HISTORY_WINDOW_DAYS, evaluate it only against
# the newer TEST fraction — never touched during generation. NOTE: despite
# the _PCT suffix (kept as-is — an env var rename risks silently dropping a
# value someone already has set), this is a 0-1 fraction, not 0-100 like
# every other _PCT constant in this file.
ADAPTIVE_TRAIN_TEST_SPLIT_PCT = float(os.getenv("ADAPTIVE_TRAIN_TEST_SPLIT_PCT", "0.7"))
# p-value threshold (two-sample z-test, normal approximation) below which a
# simulated improvement counts as statistically significant, not noise.
SIGNIFICANCE_THRESHOLD = float(os.getenv("SIGNIFICANCE_THRESHOLD", "0.05"))

# Adaptive confidence chain: base (AI+historical, unchanged) + regime +
# symbol + recent-performance modifiers, each capped and gated on sample
# size. Regime/symbol share one formula and one pair of constants since
# both are "this bucket's win rate vs. the overall baseline".
BUCKET_MODIFIER_SENSITIVITY = float(os.getenv("BUCKET_MODIFIER_SENSITIVITY", "20"))
BUCKET_MODIFIER_CAP = float(os.getenv("BUCKET_MODIFIER_CAP", "10"))
RECENT_PERFORMANCE_LOOKBACK_TRADES = int(os.getenv("RECENT_PERFORMANCE_LOOKBACK_TRADES", "10"))
# Deliberately asymmetric defaults — a losing streak can suppress
# confidence more than a winning streak inflates it (standard
# risk-management practice), but both are independently configurable.
RECENT_STREAK_WIN_MODIFIER_CAP = float(os.getenv("RECENT_STREAK_WIN_MODIFIER_CAP", "5"))
RECENT_STREAK_LOSS_MODIFIER_CAP = float(os.getenv("RECENT_STREAK_LOSS_MODIFIER_CAP", "15"))

# --- Event-Driven Backtesting & Walk-Forward Validation Engine --------------
# Replays real historical OHLCV (CoinDCX's public candles endpoint supports
# startTime/endTime even though src/coindcx_client.py's live wrapper doesn't
# expose them) through the same pure pipeline functions live trading uses
# (feature engine, opportunity scorer, risk manager) — see PROJECT_SPEC.md
# §3c. On-demand CLI only (python -m src.backtest.engine), not a cron job.

# Finest granularity the SimulationClock ticks at — governs candle
# visibility/no-look-ahead ONLY, not how often scoring/risk logic fires
# (that's the two cadences below, kept deliberately separate). Must be one
# of FEATURE_TIMEFRAMES.
BACKTEST_TICK_TIMEFRAME = os.getenv("BACKTEST_TICK_TIMEFRAME") or "1m"
# Mirrors trading_cycle.yml's */10 cron — when the full scoring+risk+entry/
# exit pass fires.
BACKTEST_DECISION_CYCLE_MINUTES = int(os.getenv("BACKTEST_DECISION_CYCLE_MINUTES", "10"))
# Mirrors risk_check.yml's */5 cron — when the stop-loss/take-profit sweep
# fires, independent of the decision cycle above (same polling-gap
# limitation as live: not continuous, see PROJECT_SPEC.md §2).
BACKTEST_RISK_CHECK_MINUTES = int(os.getenv("BACKTEST_RISK_CHECK_MINUTES", "5"))
# Historical candles are ingested from (start_date - this many days) so
# FEATURE_CANDLE_LIMIT/EMA_TREND_PERIOD_4's ~200-bar warm-up requirement is
# satisfied BEFORE the requested backtest window starts — otherwise every
# run would silently find zero candidates for its first ~200 days.
BACKTEST_WARMUP_BUFFER_DAYS = int(os.getenv("BACKTEST_WARMUP_BUFFER_DAYS", "260"))

BACKTEST_STARTING_CAPITAL = float(os.getenv("BACKTEST_STARTING_CAPITAL", "100000"))
BACKTEST_POSITION_SIZE_PCT = float(os.getenv("BACKTEST_POSITION_SIZE_PCT", "10"))
BACKTEST_MAX_CONCURRENT_POSITIONS = int(os.getenv("BACKTEST_MAX_CONCURRENT_POSITIONS", "5"))

# Execution simulation. Independent of execution/paper.py's own hardcoded
# SLIPPAGE_BPS (that module's live behavior is untouched) — commission
# reuses paper.py's exact fee formula directly, everything else here is new.
BACKTEST_SLIPPAGE_BPS = float(os.getenv("BACKTEST_SLIPPAGE_BPS", "5"))
# Synthetic half-spread cost — CoinDCX exposes no historical order-book
# snapshots (get_orderbook is live-only, never called historically), so
# real book-depth replay isn't possible; this is a documented approximation.
BACKTEST_SPREAD_BPS = float(os.getenv("BACKTEST_SPREAD_BPS", "10"))
BACKTEST_MAX_FILL_PCT_OF_BAR_VOLUME = float(os.getenv("BACKTEST_MAX_FILL_PCT_OF_BAR_VOLUME", "10"))
BACKTEST_ORDER_EXPIRY_BARS = int(os.getenv("BACKTEST_ORDER_EXPIRY_BARS", "20"))
# Mirrors CoinDCX's real ~₹100 min_notional (see README's real-execution
# caveat) as a rejection floor for simulated orders.
BACKTEST_MIN_NOTIONAL_INR = float(os.getenv("BACKTEST_MIN_NOTIONAL_INR", "100"))

# All resampling (bootstrap CI, Monte Carlo trade-order shuffling) draws
# from a local random.Random(BACKTEST_RANDOM_SEED) instance, never the
# global `random` module — reruns are bit-identical, satisfying
# "everything must be deterministic."
BACKTEST_RANDOM_SEED = int(os.getenv("BACKTEST_RANDOM_SEED", "42"))
BACKTEST_BOOTSTRAP_ITERATIONS = int(os.getenv("BACKTEST_BOOTSTRAP_ITERATIONS", "1000"))
BACKTEST_MONTE_CARLO_ITERATIONS = int(os.getenv("BACKTEST_MONTE_CARLO_ITERATIONS", "1000"))

BACKTEST_WALK_FORWARD_N_FOLDS = int(os.getenv("BACKTEST_WALK_FORWARD_N_FOLDS", "5"))
BACKTEST_WALK_FORWARD_TRAIN_DAYS = int(os.getenv("BACKTEST_WALK_FORWARD_TRAIN_DAYS", "90"))
BACKTEST_WALK_FORWARD_TEST_DAYS = int(os.getenv("BACKTEST_WALK_FORWARD_TEST_DAYS", "30"))
# RECOMMENDATION_MIN_SAMPLE_SIZE and SIGNIFICANCE_THRESHOLD (above) are
# reused directly for the fold-sample-size gate and pass/fail checks — no
# duplicate constants. Below that per-fold sample floor, folds report
# "insufficient sample" (None) rather than a hand-rolled Student's-t
# p-value — this codebase has zero numpy/scipy, and a parametric t-test
# needs a regularized-incomplete-beta implementation that's real numerical
# bug surface for little gain over the existing z-test at this sample size;
# small-n confidence intervals are answered by the seeded bootstrap above
# instead, which needs no distributional assumption at all.

# Off by default: quant-only (feature engine + opportunity scorer + risk
# manager) is the deterministic default that actually satisfies "everything
# must be deterministic" — the live LLM signal agent is temperature-sampled.
# When enabled, validate_opportunity() is reused as-is for realism, but the
# historical-confidence/regime/symbol blend is deliberately NOT reused (it
# queries LIVE current trades/learning_statistics, which would leak
# present-day trade history into a historical decision) — LLM-mode
# confidence is the raw AI verdict only, and is labeled non-reproducible in
# reports rather than fed into PerformanceAnalyzer's trusted default metrics.
BACKTEST_USE_LLM_SIGNAL_AGENT = (os.getenv("BACKTEST_USE_LLM_SIGNAL_AGENT") or "false").strip().lower() == "true"

# CoinDCX's public candles endpoint caps at 500 rows per call regardless of
# the requested startTime/endTime range (confirmed empirically) — named
# here so pagination logic has no magic number.
BACKTEST_CANDLE_PAGE_SIZE = int(os.getenv("BACKTEST_CANDLE_PAGE_SIZE", "500"))

# --- Market Data Quality Engine + Data Repair Engine ------------------------
# src/data_quality/validator.py + repair.py. One shared entry point for both
# live (data_agent.py, right after get_candles()) and backtest (data_provider
# .py::ingest, once at ingest time) — see PROJECT_SPEC.md §3d. Each check
# maps to a severity; "reject" drops the offending candle(s) from what
# reaches the Feature Engine, "quarantine" drops the whole symbol for that
# fetch, "warn" logs but passes through, "ignore" doesn't even log.
DATA_QUALITY_SEVERITY_MISSING_CANDLE = os.getenv("DATA_QUALITY_SEVERITY_MISSING_CANDLE") or "warn"
DATA_QUALITY_SEVERITY_DUPLICATE = os.getenv("DATA_QUALITY_SEVERITY_DUPLICATE") or "warn"
DATA_QUALITY_SEVERITY_NEGATIVE_PRICE = os.getenv("DATA_QUALITY_SEVERITY_NEGATIVE_PRICE") or "reject"
DATA_QUALITY_SEVERITY_INVALID_OHLC = os.getenv("DATA_QUALITY_SEVERITY_INVALID_OHLC") or "reject"
DATA_QUALITY_SEVERITY_OUT_OF_ORDER = os.getenv("DATA_QUALITY_SEVERITY_OUT_OF_ORDER") or "warn"
DATA_QUALITY_SEVERITY_TIMESTAMP_GAP = os.getenv("DATA_QUALITY_SEVERITY_TIMESTAMP_GAP") or "warn"
DATA_QUALITY_SEVERITY_ZERO_VOLUME = os.getenv("DATA_QUALITY_SEVERITY_ZERO_VOLUME") or "warn"
DATA_QUALITY_SEVERITY_PRICE_SPIKE = os.getenv("DATA_QUALITY_SEVERITY_PRICE_SPIKE") or "warn"
DATA_QUALITY_SEVERITY_EXCHANGE_OUTAGE = os.getenv("DATA_QUALITY_SEVERITY_EXCHANGE_OUTAGE") or "reject"
DATA_QUALITY_SEVERITY_CLOCK_DRIFT = os.getenv("DATA_QUALITY_SEVERITY_CLOCK_DRIFT") or "warn"
DATA_QUALITY_SEVERITY_SYMBOL_MISMATCH = os.getenv("DATA_QUALITY_SEVERITY_SYMBOL_MISMATCH") or "reject"
DATA_QUALITY_SEVERITY_TIMEFRAME_CHANGE = os.getenv("DATA_QUALITY_SEVERITY_TIMEFRAME_CHANGE") or "warn"
# Pct jump vs. the prior close (same-timeframe) beyond which a candle counts
# as an extreme spike, not ordinary volatility.
DATA_QUALITY_PRICE_SPIKE_PCT_THRESHOLD = float(os.getenv("DATA_QUALITY_PRICE_SPIKE_PCT_THRESHOLD", "20"))
# Live-fetch path only (candle time vs. wall clock) — meaningless on
# historical/backtest data, which is never "now".
DATA_QUALITY_CLOCK_DRIFT_SECONDS_THRESHOLD = int(
    os.getenv("DATA_QUALITY_CLOCK_DRIFT_SECONDS_THRESHOLD", "300")
)
# A gap this many bars or fewer gets linearly interpolated by the repair
# engine; wider gaps are left rejected — repairing a large hole would
# fabricate market data, not recover from a blip.
DATA_REPAIR_MAX_GAP_BARS = int(os.getenv("DATA_REPAIR_MAX_GAP_BARS", "3"))

# --- Portfolio Intelligence Engine ------------------------------------------
# src/portfolio/intelligence.py. Pure functions — no DB/network access; the
# caller (live risk_manager or the backtest engine) supplies positions and
# an already-as-of-truncated price_history. See PROJECT_SPEC.md §3d.

# "SYMBOL:category,SYMBOL:category" — no external crypto-category taxonomy
# exists for this bot to query, so this is a manually maintained mapping.
# Unmapped symbols fall into "uncategorized", never a guessed category.
COIN_CATEGORY_MAP = {
    sym.strip().upper(): category.strip()
    for sym, category in (
        pair.split(":") for pair in (os.getenv("COIN_CATEGORY_MAP") or "").split(",") if pair.strip()
    )
}
PORTFOLIO_VAR_CONFIDENCE_PCT = float(os.getenv("PORTFOLIO_VAR_CONFIDENCE_PCT", "95"))
# Rolling-correlation / beta window width, in bars of whatever price_history
# the caller supplied (daily closes for a live snapshot, backtest ticks for
# a backtest run) — not a separate data fetch.
PORTFOLIO_CORRELATION_LOOKBACK_BARS = int(os.getenv("PORTFOLIO_CORRELATION_LOOKBACK_BARS", "30"))
# Market proxy for beta regression — no broad crypto index exists on
# CoinDCX, BTC is the closest thing to a market benchmark.
PORTFOLIO_BETA_PROXY_SYMBOL = os.getenv("PORTFOLIO_BETA_PROXY_SYMBOL") or "BTCINR"
# Concentration caps scale with capital_config.max_concurrent_positions
# rather than a fixed institutional-style percentage: an equal-weighted
# N-position book has each position at 1/N already, so a flat 25% cap
# would block nearly every trade in this bot's actual 2-5-position range
# (a real bug a test caught — the very first position in a 2-slot book is
# structurally 100% concentrated, which isn't "too concentrated", it's
# just what one position looks like). "Equal share" = 100/max_concurrent_
# positions; the cap is that share times this multiple.
MAX_POSITION_CONCENTRATION_MULT_OF_EQUAL_SHARE = float(
    os.getenv("MAX_POSITION_CONCENTRATION_MULT_OF_EQUAL_SHARE", "2.0")
)
MAX_SECTOR_CONCENTRATION_MULT_OF_EQUAL_SHARE = float(
    os.getenv("MAX_SECTOR_CONCENTRATION_MULT_OF_EQUAL_SHARE", "2.5")
)

# --- Capital Allocation Engine ----------------------------------------------
# src/portfolio/capital_allocation.py. Only used when capital_config.
# sizing_mode='dynamic' — the DB row default is 'flat' (today's exact flat
# capital_to_use*position_size_pct/100 formula, byte-identical), so this
# whole block is inert until a human flips a mode's row in Supabase. Each
# factor is an independent multiplier clamped to its own MIN/MAX, and the
# combined product is clamped again — no single factor or their product can
# blow sizing out unboundedly. See PROJECT_SPEC.md §3d.
CAPITAL_ALLOC_CORRELATION_MIN_MULT = float(os.getenv("CAPITAL_ALLOC_CORRELATION_MIN_MULT", "0.5"))
CAPITAL_ALLOC_CORRELATION_MAX_MULT = float(os.getenv("CAPITAL_ALLOC_CORRELATION_MAX_MULT", "1.5"))
CAPITAL_ALLOC_VOLATILITY_MIN_MULT = float(os.getenv("CAPITAL_ALLOC_VOLATILITY_MIN_MULT", "0.5"))
CAPITAL_ALLOC_VOLATILITY_MAX_MULT = float(os.getenv("CAPITAL_ALLOC_VOLATILITY_MAX_MULT", "1.5"))
CAPITAL_ALLOC_DRAWDOWN_MIN_MULT = float(os.getenv("CAPITAL_ALLOC_DRAWDOWN_MIN_MULT", "0.5"))
CAPITAL_ALLOC_DRAWDOWN_MAX_MULT = float(os.getenv("CAPITAL_ALLOC_DRAWDOWN_MAX_MULT", "1.5"))
CAPITAL_ALLOC_EXPOSURE_MIN_MULT = float(os.getenv("CAPITAL_ALLOC_EXPOSURE_MIN_MULT", "0.5"))
CAPITAL_ALLOC_EXPOSURE_MAX_MULT = float(os.getenv("CAPITAL_ALLOC_EXPOSURE_MAX_MULT", "1.5"))
CAPITAL_ALLOC_STRATEGY_PERFORMANCE_MIN_MULT = float(
    os.getenv("CAPITAL_ALLOC_STRATEGY_PERFORMANCE_MIN_MULT", "0.5")
)
CAPITAL_ALLOC_STRATEGY_PERFORMANCE_MAX_MULT = float(
    os.getenv("CAPITAL_ALLOC_STRATEGY_PERFORMANCE_MAX_MULT", "1.5")
)
CAPITAL_ALLOC_REGIME_MIN_MULT = float(os.getenv("CAPITAL_ALLOC_REGIME_MIN_MULT", "0.5"))
CAPITAL_ALLOC_REGIME_MAX_MULT = float(os.getenv("CAPITAL_ALLOC_REGIME_MAX_MULT", "1.5"))
CAPITAL_ALLOC_CONFIDENCE_MIN_MULT = float(os.getenv("CAPITAL_ALLOC_CONFIDENCE_MIN_MULT", "0.5"))
CAPITAL_ALLOC_CONFIDENCE_MAX_MULT = float(os.getenv("CAPITAL_ALLOC_CONFIDENCE_MAX_MULT", "1.5"))
# Clamp on the product of every factor above, the final safety rail before
# the result still has to clear risk_manager's existing committed_capital
# ceiling unchanged.
CAPITAL_ALLOC_TOTAL_MIN_MULT = float(os.getenv("CAPITAL_ALLOC_TOTAL_MIN_MULT", "0.5"))
CAPITAL_ALLOC_TOTAL_MAX_MULT = float(os.getenv("CAPITAL_ALLOC_TOTAL_MAX_MULT", "1.5"))

# --- Fees ---------------------------------------------------------------------
# CoinDCX spot: 0.5% trading fee on trade value (both sides), +18% GST on
# that fee. Sells additionally carry 1% TDS (Income Tax Act s.194S) on trade
# value — a separate tax deduction, not something GST applies to, and not
# charged on buys (no "transfer" of the asset on acquisition). Shared by
# src/agents/execution/paper.py (simulates the fee) and real.py (adds TDS on
# top of the exchange's own reported fee_amount, which already includes GST).
TRADING_FEE_PCT = float(os.getenv("TRADING_FEE_PCT", "0.5"))
GST_PCT_ON_FEE = float(os.getenv("GST_PCT_ON_FEE", "18"))
SELL_TDS_PCT = float(os.getenv("SELL_TDS_PCT", "1"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "5"))

# --- Execution Optimizer ----------------------------------------------------
# src/execution_optimizer/optimizer.py. Real trades: recommendation is
# computed and audit-logged only, RealExecutionAgent stays market-only
# (unverified/inert, see PROJECT_SPEC.md §2). Paper trades: config-gated
# opt-in to actually act on a recommended order type via the existing
# backtest execution_simulator's fill logic (paper is simulated — safe to
# exercise). Off by default.
EXECUTION_OPTIMIZER_ENABLED = (os.getenv("EXECUTION_OPTIMIZER_ENABLED") or "false").strip().lower() == "true"
# Spread above this favors a limit order (avoid crossing a wide synthetic
# spread) over a market order, when the estimated fill probability clears
# the floor below.
EXECUTION_OPTIMIZER_SPREAD_BPS_LIMIT_THRESHOLD = float(
    os.getenv("EXECUTION_OPTIMIZER_SPREAD_BPS_LIMIT_THRESHOLD", "15")
)
EXECUTION_OPTIMIZER_MIN_FILL_PROBABILITY = float(
    os.getenv("EXECUTION_OPTIMIZER_MIN_FILL_PROBABILITY", "0.7")
)

# --- Feature Drift Detection -------------------------------------------------
# src/learning/drift_detection.py. Runs as its own independent nightly step
# in evolution.yml (never merged into evolution_agent.run_evolution() or
# adaptive_strategy_engine — same "don't couple independent learning steps"
# rule those two already follow). Advisory only, writes to drift_alerts;
# nothing here touches config.py or scoring weights.
DRIFT_BASELINE_WINDOW_DAYS = int(os.getenv("DRIFT_BASELINE_WINDOW_DAYS", "90"))
DRIFT_RECENT_WINDOW_DAYS = int(os.getenv("DRIFT_RECENT_WINDOW_DAYS", "14"))
# Population Stability Index (hand-rolled bucketed frequency ratio, no
# scipy) — <0.1 stable, 0.1-0.25 warning, >=0.25 critical, an industry-
# standard convention for this metric.
DRIFT_PSI_WARNING_THRESHOLD = float(os.getenv("DRIFT_PSI_WARNING_THRESHOLD", "0.1"))
DRIFT_PSI_CRITICAL_THRESHOLD = float(os.getenv("DRIFT_PSI_CRITICAL_THRESHOLD", "0.25"))
DRIFT_PSI_BUCKET_COUNT = int(os.getenv("DRIFT_PSI_BUCKET_COUNT", "10"))

# --- Strategy Health Engine --------------------------------------------------
# src/learning/strategy_health.py. Health-score tiers, checked in descending
# order (>=EXCELLENT -> Excellent, >=GOOD -> Good, >=WARNING -> Warning,
# else Critical). Auto-suspension marks strategy_versions.status='suspended'
# only (never a delete) — reversible in Supabase at any time.
STRATEGY_HEALTH_EXCELLENT_THRESHOLD = float(os.getenv("STRATEGY_HEALTH_EXCELLENT_THRESHOLD", "80"))
STRATEGY_HEALTH_GOOD_THRESHOLD = float(os.getenv("STRATEGY_HEALTH_GOOD_THRESHOLD", "60"))
STRATEGY_HEALTH_WARNING_THRESHOLD = float(os.getenv("STRATEGY_HEALTH_WARNING_THRESHOLD", "40"))
STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED = (
    os.getenv("STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED") or "true"
).strip().lower() == "true"
# RECOMMENDATION_MIN_SAMPLE_SIZE (above) is reused directly as the "trust
# this health score" trade-count floor — no duplicate constant.

# --- Production Monitoring & Self-Diagnostics -------------------------------
# src/monitoring/. Scoped to what's real for short-lived GitHub Actions cron
# invocations (stateless, Supabase as the only durable state) rather than
# invented long-running-server metaphors. diagnostics runs as a new step in
# risk_check.yml (every 5 min, already the finest-grained cron) — no new
# workflow file. See PROJECT_SPEC.md §3d.
SYSTEM_METRICS_MARKET_FEED_STALE_MINUTES = int(os.getenv("SYSTEM_METRICS_MARKET_FEED_STALE_MINUTES", "30"))
SYSTEM_METRICS_LEARNING_STALE_HOURS = int(os.getenv("SYSTEM_METRICS_LEARNING_STALE_HOURS", "48"))

# --- Resilience --------------------------------------------------------------
# src/resilience.py. retry_with_backoff() wraps coindcx_client.py's requests
# calls and db/models.py's Supabase calls (both zero-retry before this) —
# same backoff shape groq_client.py already used for LLM calls, now the one
# shared implementation. The circuit breaker is DB-backed (survives across
# cron invocations) and fails OPEN on its own write errors — a Supabase
# outage can't block itself from being recorded as a Supabase outage.
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_BASE_DELAY_SECONDS = float(os.getenv("RETRY_BASE_DELAY_SECONDS", "1"))
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
CIRCUIT_BREAKER_COOLDOWN_SECONDS = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "300"))
# groq_client.py's per-model retry chain — same backoff_delay() formula as
# retry_with_backoff above, a separate attempt count/base delay since it's a
# genuinely different concern (retries before falling to the next model in
# the chain, not a flat retry).
LLM_MAX_RETRIES_PER_MODEL = int(os.getenv("LLM_MAX_RETRIES_PER_MODEL", "2"))
LLM_BACKOFF_BASE_SECONDS = float(os.getenv("LLM_BACKOFF_BASE_SECONDS", "1.0"))

# --- Scientific Strategy Optimization ----------------------------------------
# src/learning/fitness.py + recommendations.py/simulation.py's extended
# candidate pipeline, replacing evolution_agent.py's retired nightly LLM
# prompt/param rewrite. Fitness weights default to the reference blend
# (30% profit factor / 25% Sharpe / 20% expectancy / 15% win rate / 10%
# drawdown penalty); renormalized among whatever components are available
# (weighted_average's existing convention), so a missing component never
# skews the score.
FITNESS_WEIGHT_PROFIT_FACTOR = float(os.getenv("FITNESS_WEIGHT_PROFIT_FACTOR", "0.30"))
FITNESS_WEIGHT_SHARPE = float(os.getenv("FITNESS_WEIGHT_SHARPE", "0.25"))
FITNESS_WEIGHT_EXPECTANCY = float(os.getenv("FITNESS_WEIGHT_EXPECTANCY", "0.20"))
FITNESS_WEIGHT_WIN_RATE = float(os.getenv("FITNESS_WEIGHT_WIN_RATE", "0.15"))
FITNESS_WEIGHT_DRAWDOWN_PENALTY = float(os.getenv("FITNESS_WEIGHT_DRAWDOWN_PENALTY", "0.10"))
# Sensitivity anchor mapping expectancy (as % of capital_to_use) onto the
# 0-100 component scale: expectancy_pct * this value is added to a neutral
# 50 baseline, clamped to [0, 100].
FITNESS_EXPECTANCY_SCALE = float(os.getenv("FITNESS_EXPECTANCY_SCALE", "10"))
# Gate for strategy_versions.promotion_eligible, alongside the existing
# PROMOTION_MIN_PAPER_DAYS/_CUMULATIVE_PNL/_MAX_DRAWDOWN_PCT thresholds.
PROMOTION_MIN_FITNESS_SCORE = float(os.getenv("PROMOTION_MIN_FITNESS_SCORE", "60"))
# stop_loss_pct/take_profit_pct candidate sweep range (recommendations.py's
# generate_exit_params_recommendations) — decimal fractions of entry price,
# same convention as params_json.stop_loss_pct/take_profit_pct itself.
EXIT_PARAM_SWEEP_MIN_PCT = float(os.getenv("EXIT_PARAM_SWEEP_MIN_PCT", "0.01"))
EXIT_PARAM_SWEEP_MAX_PCT = float(os.getenv("EXIT_PARAM_SWEEP_MAX_PCT", "0.10"))
EXIT_PARAM_SWEEP_STEP_PCT = float(os.getenv("EXIT_PARAM_SWEEP_STEP_PCT", "0.01"))

# --- Progressive Learning Stages ----------------------------------------------
# Replaces RECOMMENDATION_MIN_SAMPLE_SIZE as the OVERALL "enough evidence
# collected to attempt this artifact type at all" gate for each layer of the
# learning engine (src/learning/learning_status.py computes which stage a
# mode is in from these same 4 boundaries). RECOMMENDATION_MIN_SAMPLE_SIZE
# itself is unchanged and keeps its original, narrower meaning: a per-bucket/
# per-subset credibility floor (e.g. "does this one symbol/regime bucket, or
# this one train/test half, have enough trades to trust its own stats") —
# that check stays exactly as strict as before at every site it already
# gates; only the single outer "is there enough evidence overall" check per
# generator moves to these staged, artifact-specific values.
LEARNING_STAGE_OBSERVATION_MIN_TRADES = int(os.getenv("LEARNING_STAGE_OBSERVATION_MIN_TRADES", "25"))
LEARNING_FEATURE_IMPORTANCE_MIN_TRADES = int(os.getenv("LEARNING_FEATURE_IMPORTANCE_MIN_TRADES", "50"))
LEARNING_STAGE_HYPOTHESIS_MIN_TRADES = int(os.getenv("LEARNING_STAGE_HYPOTHESIS_MIN_TRADES", "100"))
LEARNING_STAGE_SIMULATION_MIN_TRADES = int(os.getenv("LEARNING_STAGE_SIMULATION_MIN_TRADES", "250"))
LEARNING_STAGE_VALIDATION_MIN_TRADES = int(os.getenv("LEARNING_STAGE_VALIDATION_MIN_TRADES", "500"))
