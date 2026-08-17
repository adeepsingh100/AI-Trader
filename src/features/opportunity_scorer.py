"""Turns Feature Engine output into a deterministic 0-100 opportunity
score. No AI, no LLM, no randomness — every mapping here is a fixed
formula over named config constants (src/config.py). Never fetches data
or makes an LLM call; consumes compute_multi_timeframe_features() output.

Missing indicators (short history) degrade gracefully: every aggregation
step (timeframe components -> a timeframe's sub-score, per-timeframe
sub-scores -> the blended sub-score, sub-scores -> the final opportunity
score) goes through the same weighted_average helper, which renormalizes
weights among whatever's actually available and returns None only when
nothing at all is available — never a fabricated 0."""

from __future__ import annotations

from src.config import (
    MIN_OPPORTUNITY_SCORE,
    OPPORTUNITY_WEIGHT_MOMENTUM,
    OPPORTUNITY_WEIGHT_RISK,
    OPPORTUNITY_WEIGHT_TREND,
    OPPORTUNITY_WEIGHT_VOLATILITY,
    OPPORTUNITY_WEIGHT_VOLUME,
    REGIME_ADX_TREND_THRESHOLD,
    REGIME_STRONG_TREND_SCORE_MIN,
    RISK_RESISTANCE_DISTANCE_FOR_MAX_SCORE,
    RSI_SCORE_CEIL,
    RSI_SCORE_FLOOR,
    STOCH_RSI_SCORE_CEIL,
    STOCH_RSI_SCORE_FLOOR,
    TIMEFRAME_WEIGHTS,
    TOP_N_CANDIDATES,
    VOLATILITY_SCORE_EXTREME,
    VOLUME_SCORE_SCALE,
)
from src.utils import clamp

# Same timeframe score_opportunity() itself trusts most (highest configured
# weight) — used for the point-in-time context (ADX/volatility_regime)
# regime classification needs but can't blend across timeframes.
PRIMARY_TIMEFRAME = max(TIMEFRAME_WEIGHTS, key=TIMEFRAME_WEIGHTS.get)


def _linear_score(value: float | None, floor: float, ceil: float) -> float | None:
    if value is None:
        return None
    return clamp((value - floor) / (ceil - floor) * 100, 0, 100)


def _bool_score(condition: bool | None) -> float | None:
    if condition is None:
        return None
    return 100.0 if condition else 0.0


def weighted_average(values: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """Weighted average of whatever's not None, renormalized among the
    available subset. None only if literally nothing is available — the
    one rule every aggregation step in this module goes through."""
    total, total_weight = 0.0, 0.0
    for key, value in values.items():
        if value is None:
            continue
        weight = weights.get(key, 0.0)
        total += value * weight
        total_weight += weight
    return total / total_weight if total_weight else None


# --- per-timeframe sub-scores ------------------------------------------


def _score_trend_for_timeframe(f: dict) -> float | None:
    close, e20, e50, e100, e200 = f["close"], f["ema_20"], f["ema_50"], f["ema_100"], f["ema_200"]
    components = {
        "close_gt_ema20": _bool_score(close > e20) if close is not None and e20 is not None else None,
        "ema20_gt_ema50": _bool_score(e20 > e50) if e20 is not None and e50 is not None else None,
        "ema50_gt_ema100": _bool_score(e50 > e100) if e50 is not None and e100 is not None else None,
        "ema100_gt_ema200": _bool_score(e100 > e200) if e100 is not None and e200 is not None else None,
    }
    return weighted_average(components, dict.fromkeys(components, 0.25))


def _score_momentum_for_timeframe(f: dict) -> float | None:
    components = {
        "rsi": _linear_score(f["rsi"], RSI_SCORE_FLOOR, RSI_SCORE_CEIL),
        "macd": _bool_score(f["macd_histogram"] > 0) if f["macd_histogram"] is not None else None,
        "stoch_rsi": _linear_score(f["stoch_rsi_k"], STOCH_RSI_SCORE_FLOOR, STOCH_RSI_SCORE_CEIL),
    }
    return weighted_average(components, dict.fromkeys(components, 1 / 3))


def _score_volume_for_timeframe(f: dict) -> float | None:
    rel_vol = f["relative_volume"]
    components = {
        "relative_volume": clamp((rel_vol - 1) * VOLUME_SCORE_SCALE, 0, 100) if rel_vol is not None else None,
        "obv": _bool_score(f["obv_rising"]),
    }
    return weighted_average(components, {"relative_volume": 0.5, "obv": 0.5})


def _score_volatility_for_timeframe(f: dict) -> float | None:
    regime = f["volatility_regime"]
    if regime is None:
        return None
    return 100.0 if regime == "medium" else VOLATILITY_SCORE_EXTREME


def _score_risk_for_timeframe(f: dict) -> float | None:
    distance = f["distance_from_resistance_pct"]
    if distance is None:
        return None
    return clamp(distance / RISK_RESISTANCE_DISTANCE_FOR_MAX_SCORE * 100, 0, 100)


# --- blended across timeframes ------------------------------------------


def _blend_across_timeframes(per_tf_scores: dict[str, float | None]) -> float | None:
    return weighted_average(per_tf_scores, TIMEFRAME_WEIGHTS)


def score_trend(features_by_tf: dict[str, dict]) -> float | None:
    return _blend_across_timeframes({tf: _score_trend_for_timeframe(f) for tf, f in features_by_tf.items()})


def score_momentum(features_by_tf: dict[str, dict]) -> float | None:
    return _blend_across_timeframes({tf: _score_momentum_for_timeframe(f) for tf, f in features_by_tf.items()})


def score_volume(features_by_tf: dict[str, dict]) -> float | None:
    return _blend_across_timeframes({tf: _score_volume_for_timeframe(f) for tf, f in features_by_tf.items()})


def score_volatility(features_by_tf: dict[str, dict]) -> float | None:
    return _blend_across_timeframes({tf: _score_volatility_for_timeframe(f) for tf, f in features_by_tf.items()})


def score_risk(features_by_tf: dict[str, dict]) -> float | None:
    return _blend_across_timeframes({tf: _score_risk_for_timeframe(f) for tf, f in features_by_tf.items()})


def classify_market_regime(features_by_tf: dict[str, dict]) -> str | None:
    """Deterministic composite label, reusing score_trend() (already
    computed) plus the primary timeframe's ADX/volatility_regime — not a
    separate indicator pass. None if either input is unavailable (short
    history), never a guessed label.

    sideways: ADX below REGIME_ADX_TREND_THRESHOLD (no real trend to
    classify direction of). high_volatility: overrides the trend label
    when the primary timeframe's volatility bucket is "high" — an
    unpredictable market matters more than its direction. Otherwise
    strong/weak bull or bear, split symmetrically around trend_score=50
    (neutral) by REGIME_STRONG_TREND_SCORE_MIN and its mirror. No separate
    "trending" label — that's what strong_bull/strong_bear already encode.
    """
    trend = score_trend(features_by_tf)
    primary = features_by_tf.get(PRIMARY_TIMEFRAME) or {}
    adx = primary.get("adx")
    if trend is None or adx is None:
        return None

    if adx < REGIME_ADX_TREND_THRESHOLD:
        return "sideways"
    if primary.get("volatility_regime") == "high":
        return "high_volatility"
    if trend >= REGIME_STRONG_TREND_SCORE_MIN:
        return "strong_bull"
    if trend >= 50:
        return "weak_bull"
    if trend <= 100 - REGIME_STRONG_TREND_SCORE_MIN:
        return "strong_bear"
    return "weak_bear"


def score_opportunity(features_by_tf: dict[str, dict]) -> dict:
    sub_scores = {
        "trend": score_trend(features_by_tf),
        "momentum": score_momentum(features_by_tf),
        "volume": score_volume(features_by_tf),
        "volatility": score_volatility(features_by_tf),
        "risk": score_risk(features_by_tf),
    }
    weights = {
        "trend": OPPORTUNITY_WEIGHT_TREND,
        "momentum": OPPORTUNITY_WEIGHT_MOMENTUM,
        "volume": OPPORTUNITY_WEIGHT_VOLUME,
        "volatility": OPPORTUNITY_WEIGHT_VOLATILITY,
        "risk": OPPORTUNITY_WEIGHT_RISK,
    }
    return {
        "trend_score": sub_scores["trend"],
        "momentum_score": sub_scores["momentum"],
        "volume_score": sub_scores["volume"],
        "volatility_score": sub_scores["volatility"],
        "risk_score": sub_scores["risk"],
        "opportunity_score": weighted_average(sub_scores, weights),
        "market_regime": classify_market_regime(features_by_tf),
    }


def select_top_candidates(
    scored: list[dict], top_n: int = TOP_N_CANDIDATES, min_score: float = MIN_OPPORTUNITY_SCORE
) -> list[dict]:
    eligible = [
        s for s in scored if s.get("opportunity_score") is not None and s["opportunity_score"] >= min_score
    ]
    eligible.sort(key=lambda s: s["opportunity_score"], reverse=True)
    return eligible[:top_n]
