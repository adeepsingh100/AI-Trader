"""Capital Allocation Engine — dynamic position sizing as a function of
portfolio risk/correlation/volatility/drawdown/exposure/strategy
performance/regime/confidence, replacing risk_manager's flat
capital_to_use*position_size_pct/100 formula. See PROJECT_SPEC.md §3d.

**Inert by default.** Only reached when capital_config.sizing_mode ==
'dynamic' (migration default is 'flat', today's exact existing formula,
byte-identical) — a human flips a mode's row in Supabase after reviewing
behavior in paper first, nothing in code does that automatically. The
result of compute_dynamic_size() still flows through risk_manager's
existing committed_capital(open_trades) + trade_capital > capital_to_use
ceiling unchanged — this only decides the multiplier, never bypasses the
capital cap.

Each factor is an independent multiplier clamped to its own
CAPITAL_ALLOC_<FACTOR>_MIN/MAX_MULT range; the combined product is clamped
again to CAPITAL_ALLOC_TOTAL_MIN/MAX_MULT. A factor whose input is
unavailable (None) contributes 1.0 (neutral, no adjustment) rather than
skewing sizing on missing data. Reuses existing config anchors where they
already exist (VOLATILITY_LOW_MAX_PCT/VOLATILITY_HIGH_MIN_PCT,
PROMOTION_MAX_DRAWDOWN_PCT) instead of inventing duplicate thresholds."""

from __future__ import annotations

from src.config import (
    CAPITAL_ALLOC_CONFIDENCE_MAX_MULT,
    CAPITAL_ALLOC_CONFIDENCE_MIN_MULT,
    CAPITAL_ALLOC_CORRELATION_MAX_MULT,
    CAPITAL_ALLOC_CORRELATION_MIN_MULT,
    CAPITAL_ALLOC_DRAWDOWN_MAX_MULT,
    CAPITAL_ALLOC_DRAWDOWN_MIN_MULT,
    CAPITAL_ALLOC_EXPOSURE_MAX_MULT,
    CAPITAL_ALLOC_EXPOSURE_MIN_MULT,
    CAPITAL_ALLOC_REGIME_MAX_MULT,
    CAPITAL_ALLOC_REGIME_MIN_MULT,
    CAPITAL_ALLOC_STRATEGY_PERFORMANCE_MAX_MULT,
    CAPITAL_ALLOC_STRATEGY_PERFORMANCE_MIN_MULT,
    CAPITAL_ALLOC_TOTAL_MAX_MULT,
    CAPITAL_ALLOC_TOTAL_MIN_MULT,
    CAPITAL_ALLOC_VOLATILITY_MAX_MULT,
    CAPITAL_ALLOC_VOLATILITY_MIN_MULT,
    PROMOTION_MAX_DRAWDOWN_PCT,
    VOLATILITY_HIGH_MIN_PCT,
    VOLATILITY_LOW_MAX_PCT,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _linear(value: float, low: float, high: float, mult_at_low: float, mult_at_high: float) -> float:
    """value<=low -> mult_at_low, value>=high -> mult_at_high, linear
    between. Named by POSITION (at_low/at_high), not by magnitude
    (min/max) — a prior version named these min_mult/max_mult, which
    silently inverted direction at any call site wanting a decreasing
    relationship (value low -> high mult) if a caller passed them in
    "natural" min-then-max order instead of low-then-high order; this
    naming makes the correct argument unambiguous at each call site."""
    if high == low:
        return (mult_at_low + mult_at_high) / 2
    frac = _clamp((value - low) / (high - low), 0.0, 1.0)
    return mult_at_low + frac * (mult_at_high - mult_at_low)


def correlation_factor(avg_correlation: float | None) -> float:
    """1.0 (fully correlated with the existing book, adds no diversification)
    -> MIN. -1.0 (perfect diversifier) -> MAX. None (no correlation data,
    e.g. first position or thin history) -> neutral 1.0."""
    if avg_correlation is None:
        return 1.0
    return _linear(
        avg_correlation, -1.0, 1.0, CAPITAL_ALLOC_CORRELATION_MAX_MULT, CAPITAL_ALLOC_CORRELATION_MIN_MULT
    )  # value at low(-1, diversifier) = MAX mult; value at high(1, correlated) = MIN mult


def volatility_factor(candidate_volatility_pct: float | None) -> float:
    """Reuses the Feature Engine's own low/high volatility boundary
    (VOLATILITY_LOW_MAX_PCT/VOLATILITY_HIGH_MIN_PCT) as the anchors, rather
    than a second, disconnected volatility threshold."""
    if candidate_volatility_pct is None:
        return 1.0
    return _linear(
        candidate_volatility_pct,
        VOLATILITY_LOW_MAX_PCT,
        VOLATILITY_HIGH_MIN_PCT,
        CAPITAL_ALLOC_VOLATILITY_MAX_MULT,
        CAPITAL_ALLOC_VOLATILITY_MIN_MULT,
    )


def drawdown_factor(recent_drawdown_pct: float | None) -> float:
    """Reuses PROMOTION_MAX_DRAWDOWN_PCT (the existing "acceptable max
    drawdown" bar) as the point at which sizing bottoms out, instead of a
    second drawdown threshold with no relationship to the first."""
    if recent_drawdown_pct is None:
        return 1.0
    return _linear(
        recent_drawdown_pct,
        0.0,
        PROMOTION_MAX_DRAWDOWN_PCT,
        CAPITAL_ALLOC_DRAWDOWN_MAX_MULT,
        CAPITAL_ALLOC_DRAWDOWN_MIN_MULT,
    )


def exposure_factor(current_exposure_pct: float | None) -> float:
    if current_exposure_pct is None:
        return 1.0
    return _linear(
        current_exposure_pct, 0.0, 100.0, CAPITAL_ALLOC_EXPOSURE_MAX_MULT, CAPITAL_ALLOC_EXPOSURE_MIN_MULT
    )


def strategy_performance_factor(strategy_win_rate: float | None) -> float:
    """0.3 win rate -> MIN, 0.7 -> MAX, linear between — deliberately not
    anchored at 0.5 (breakeven win rate can still be net-profitable
    depending on win/loss size, so it isn't the natural midpoint here)."""
    if strategy_win_rate is None:
        return 1.0
    return _linear(
        strategy_win_rate,
        0.3,
        0.7,
        CAPITAL_ALLOC_STRATEGY_PERFORMANCE_MIN_MULT,
        CAPITAL_ALLOC_STRATEGY_PERFORMANCE_MAX_MULT,
    )


_REGIME_FRACTION = {
    "strong_bull": 1.0,
    "weak_bull": 0.75,
    "sideways": 0.5,
    "weak_bear": 0.25,
    "strong_bear": 0.0,
    "high_volatility": 0.0,
}


def regime_factor(market_regime: str | None) -> float:
    fraction = _REGIME_FRACTION.get(market_regime)
    if fraction is None:
        return 1.0
    return CAPITAL_ALLOC_REGIME_MIN_MULT + fraction * (
        CAPITAL_ALLOC_REGIME_MAX_MULT - CAPITAL_ALLOC_REGIME_MIN_MULT
    )


def confidence_factor(confidence: float | None) -> float:
    if confidence is None:
        return 1.0
    fraction = _clamp(confidence / 100.0, 0.0, 1.0)
    return CAPITAL_ALLOC_CONFIDENCE_MIN_MULT + fraction * (
        CAPITAL_ALLOC_CONFIDENCE_MAX_MULT - CAPITAL_ALLOC_CONFIDENCE_MIN_MULT
    )


def compute_dynamic_size(
    base_trade_capital: float,
    avg_correlation: float | None = None,
    candidate_volatility_pct: float | None = None,
    recent_drawdown_pct: float | None = None,
    current_exposure_pct: float | None = None,
    strategy_win_rate: float | None = None,
    market_regime: str | None = None,
    confidence: float | None = None,
) -> dict:
    factors = {
        "correlation": correlation_factor(avg_correlation),
        "volatility": volatility_factor(candidate_volatility_pct),
        "drawdown": drawdown_factor(recent_drawdown_pct),
        "exposure": exposure_factor(current_exposure_pct),
        "strategy_performance": strategy_performance_factor(strategy_win_rate),
        "regime": regime_factor(market_regime),
        "confidence": confidence_factor(confidence),
    }
    combined = 1.0
    for f in factors.values():
        combined *= f
    combined = _clamp(combined, CAPITAL_ALLOC_TOTAL_MIN_MULT, CAPITAL_ALLOC_TOTAL_MAX_MULT)
    return {
        "trade_capital": base_trade_capital * combined,
        "combined_multiplier": combined,
        "factors": factors,
    }
