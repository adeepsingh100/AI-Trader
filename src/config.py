from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

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

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

COINDCX_API_KEY = os.getenv("COINDCX_API_KEY", "")
COINDCX_API_SECRET = os.getenv("COINDCX_API_SECRET", "")

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
RECOMMENDATION_MIN_IMPROVEMENT_PCT = float(os.getenv("RECOMMENDATION_MIN_IMPROVEMENT_PCT", "15"))
RECOMMENDATION_MIN_SAMPLE_SIZE = int(os.getenv("RECOMMENDATION_MIN_SAMPLE_SIZE", "20"))
