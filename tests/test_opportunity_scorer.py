from unittest.mock import patch

import pytest

from src.features.opportunity_scorer import (
    classify_market_regime,
    score_momentum,
    score_opportunity,
    score_risk,
    score_trend,
    score_volatility,
    score_volume,
    select_top_candidates,
)


def _features(**overrides) -> dict:
    base = {
        "close": None, "ema_20": None, "ema_50": None, "ema_100": None, "ema_200": None,
        "rsi": None, "macd_line": None, "macd_signal": None, "macd_histogram": None,
        "stoch_rsi_k": None, "stoch_rsi_d": None, "atr": None, "atr_pct": None,
        "bollinger_width_pct": None, "relative_volume": None, "volume_spike": None,
        "obv": None, "obv_rising": None, "support": None, "resistance": None,
        "distance_from_support_pct": None, "distance_from_resistance_pct": None,
        "adx": None, "di_plus": None, "di_minus": None, "volatility_regime": None,
    }
    base.update(overrides)
    return base


_WEIGHTS = {"5m": 0.15, "15m": 0.25, "1h": 0.30, "4h": 0.30}


def _by_tf(**overrides) -> dict:
    return {tf: _features(**overrides) for tf in _WEIGHTS}


# --- score_trend ---


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_score_trend_full_bullish_stack_is_100():
    features = _by_tf(close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70)
    assert score_trend(features) == 100.0


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_score_trend_full_bearish_stack_is_0():
    features = _by_tf(close=70, ema_20=80, ema_50=90, ema_100=100, ema_200=110)
    assert score_trend(features) == 0.0


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_score_trend_none_when_all_emas_missing():
    features = _by_tf(close=100)
    assert score_trend(features) is None


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_score_trend_degrades_gracefully_with_partial_emas():
    # only close>ema20 is knowable (true) -> that alone determines the score
    features = _by_tf(close=110, ema_20=100)
    assert score_trend(features) == 100.0


# --- score_momentum ---


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.RSI_SCORE_FLOOR", 30)
@patch("src.features.opportunity_scorer.RSI_SCORE_CEIL", 70)
def test_score_momentum_rsi_at_ceiling_scores_100_component():
    # rsi=70 (ceiling) + no macd/stoch data -> momentum score driven by rsi alone
    features = _by_tf(rsi=70)
    assert score_momentum(features) == pytest.approx(100.0)


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_score_momentum_macd_positive_histogram_scores_100_component():
    features = _by_tf(macd_histogram=0.5)
    assert score_momentum(features) == pytest.approx(100.0)


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_score_momentum_macd_negative_histogram_scores_0_component():
    features = _by_tf(macd_histogram=-0.5)
    assert score_momentum(features) == 0.0


# --- score_volume ---


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.VOLUME_SCORE_SCALE", 50)
def test_score_volume_relative_volume_and_obv_combine():
    # relative_volume=3 -> (3-1)*50=100 clamped; obv_rising=True -> 100
    features = _by_tf(relative_volume=3.0, obv_rising=True)
    assert score_volume(features) == 100.0


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_score_volume_none_when_nothing_available():
    features = _by_tf()
    assert score_volume(features) is None


# --- score_volatility ---


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.VOLATILITY_SCORE_EXTREME", 40)
def test_score_volatility_medium_is_best():
    assert score_volatility(_by_tf(volatility_regime="medium")) == 100.0
    assert score_volatility(_by_tf(volatility_regime="low")) == 40.0
    assert score_volatility(_by_tf(volatility_regime="high")) == 40.0


# --- score_risk ---


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.RISK_RESISTANCE_DISTANCE_FOR_MAX_SCORE", 5.0)
def test_score_risk_scales_with_distance_from_resistance():
    assert score_risk(_by_tf(distance_from_resistance_pct=5.0)) == 100.0
    assert score_risk(_by_tf(distance_from_resistance_pct=2.5)) == 50.0
    assert score_risk(_by_tf(distance_from_resistance_pct=10.0)) == 100.0  # clamped
    assert score_risk(_by_tf(distance_from_resistance_pct=0.0)) == 0.0


# --- score_opportunity: renormalization across timeframes and sub-scores ---


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", {"5m": 0.5, "1h": 0.5})
def test_score_opportunity_renormalizes_when_one_timeframe_entirely_missing():
    # 5m has full bullish trend data, 1h has nothing at all for trend inputs
    features_by_tf = {
        "5m": _features(close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70),
        "1h": _features(),
    }
    result = score_opportunity(features_by_tf)
    # only 5m contributes to trend_score -> renormalized weight of 1.0 -> 100
    assert result["trend_score"] == 100.0


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_score_opportunity_none_when_every_sub_score_is_none():
    result = score_opportunity(_by_tf())
    assert result["opportunity_score"] is None
    assert result["trend_score"] is None


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_TREND", 1.0)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_MOMENTUM", 0.0)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_VOLUME", 0.0)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_VOLATILITY", 0.0)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_RISK", 0.0)
def test_score_opportunity_weight_normalization_proven_with_extreme_weights():
    # trend_weight=1, everything else 0 -> opportunity_score == trend_score
    # exactly, proving the weighted blend actually uses configured weights
    features_by_tf = _by_tf(close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70)
    result = score_opportunity(features_by_tf)
    assert result["opportunity_score"] == result["trend_score"] == 100.0


# --- select_top_candidates ---


def test_select_top_candidates_filters_sorts_and_slices():
    scored = [
        {"symbol": "A", "opportunity_score": 90},
        {"symbol": "B", "opportunity_score": 40},  # below min_score
        {"symbol": "C", "opportunity_score": 75},
        {"symbol": "D", "opportunity_score": 60},
        {"symbol": "E", "opportunity_score": None},  # unscoreable
        {"symbol": "F", "opportunity_score": 85},
    ]
    result = select_top_candidates(scored, top_n=2, min_score=60)
    assert [r["symbol"] for r in result] == ["A", "F"]


def test_select_top_candidates_excludes_none_without_raising():
    scored = [{"symbol": "A", "opportunity_score": None}]
    assert select_top_candidates(scored, top_n=5, min_score=60) == []


def test_select_top_candidates_empty_input():
    assert select_top_candidates([], top_n=5, min_score=60) == []


# --- classify_market_regime ---


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.REGIME_ADX_TREND_THRESHOLD", 20)
def test_classify_market_regime_sideways_on_low_adx():
    features = _by_tf(close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70, adx=10, volatility_regime="low")
    assert classify_market_regime(features) == "sideways"


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.REGIME_ADX_TREND_THRESHOLD", 20)
def test_classify_market_regime_high_volatility_overrides_trend():
    features = _by_tf(close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70, adx=30, volatility_regime="high")
    assert classify_market_regime(features) == "high_volatility"


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.REGIME_ADX_TREND_THRESHOLD", 20)
@patch("src.features.opportunity_scorer.REGIME_STRONG_TREND_SCORE_MIN", 75)
def test_classify_market_regime_strong_bull_on_full_bullish_stack():
    features = _by_tf(close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70, adx=30, volatility_regime="medium")
    assert classify_market_regime(features) == "strong_bull"


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.REGIME_ADX_TREND_THRESHOLD", 20)
@patch("src.features.opportunity_scorer.REGIME_STRONG_TREND_SCORE_MIN", 75)
def test_classify_market_regime_strong_bear_on_full_bearish_stack():
    features = _by_tf(close=70, ema_20=80, ema_50=90, ema_100=100, ema_200=110, adx=30, volatility_regime="medium")
    assert classify_market_regime(features) == "strong_bear"


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_classify_market_regime_none_when_adx_unavailable():
    features = _by_tf(close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70, adx=None)
    assert classify_market_regime(features) is None


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
def test_score_opportunity_includes_market_regime():
    features = _by_tf(close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70, adx=30, volatility_regime="medium")
    result = score_opportunity(features)
    assert "market_regime" in result


# --- multi-strategy-type: explicit profile args override the module
# global, but a bare call (no args) still honors @patch on the global --
# the two halves of the "resolved inside the body, not a signature
# default" contract every parameterized function in this module follows.


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", _WEIGHTS)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_TREND", 1.0)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_MOMENTUM", 0.0)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_VOLUME", 0.0)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_VOLATILITY", 0.0)
@patch("src.features.opportunity_scorer.OPPORTUNITY_WEIGHT_RISK", 0.0)
def test_score_opportunity_explicit_profile_overrides_module_global_weights():
    # Bullish trend (high) but weak momentum (low RSI) -- default global
    # weights (all-trend) and an explicit all-momentum profile must
    # disagree on the resulting opportunity_score.
    features = _by_tf(
        close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70,  # trend = 100
        rsi=20,  # momentum pulled down
    )
    default_score = score_opportunity(features)["opportunity_score"]
    momentum_profile_score = score_opportunity(
        features, opportunity_weights={"trend": 0.0, "momentum": 1.0, "volume": 0.0, "volatility": 0.0, "risk": 0.0}
    )["opportunity_score"]
    assert default_score != pytest.approx(momentum_profile_score)


@patch("src.features.opportunity_scorer.TIMEFRAME_WEIGHTS", {"1h": 1.0})
def test_score_trend_explicit_timeframe_weights_still_overridable_when_patch_omitted():
    # No @patch on this call site -- confirms the bare call still reads
    # the (patched-by-the-class-decorator) module global, while a second
    # call in the same test can override it with an explicit dict.
    features = {"1h": _features(close=110, ema_20=100, ema_50=90, ema_100=80, ema_200=70)}
    bare = score_trend(features)
    explicit = score_trend(features, timeframe_weights={"1h": 1.0})
    assert bare == explicit == 100.0
