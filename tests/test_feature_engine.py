import pytest

from src.features.feature_engine import (
    adx,
    atr,
    bollinger_bands,
    compute_features,
    compute_multi_timeframe_features,
    ema,
    macd,
    obv,
    relative_volume,
    rsi,
    stoch_rsi,
    support_resistance,
    volatility_bucket,
    volume_spike,
)


def _candle(i, o, h, l, c, v):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v, "time": i}


# --- never raise on short history ---


def test_ema_none_on_short_history():
    assert ema([1.0, 2.0], 200) is None


def test_rsi_none_on_short_history():
    assert rsi([1.0, 2.0], period=14) is None


def test_macd_none_on_short_history():
    assert macd([1.0, 2.0], fast=12, slow=26, signal=9) is None


def test_atr_none_on_short_history():
    assert atr([1.0], [1.0], [1.0], period=14) is None


def test_bollinger_none_on_short_history():
    assert bollinger_bands([1.0, 2.0], period=20) is None


def test_relative_volume_none_on_short_history():
    assert relative_volume([1.0, 2.0], lookback=20) is None


def test_adx_none_on_short_history():
    assert adx([1.0] * 5, [1.0] * 5, [1.0] * 5, period=14) is None


def test_support_resistance_none_on_empty():
    assert support_resistance([], []) is None


def test_compute_features_on_empty_candles_returns_all_none_not_raise():
    features = compute_features([])
    assert features["close"] is None
    assert features["rsi"] is None
    assert set(features.keys()) == {
        "close", "ema_20", "ema_50", "ema_100", "ema_200", "rsi", "macd_line", "macd_signal",
        "macd_histogram", "stoch_rsi_k", "stoch_rsi_d", "atr", "atr_pct", "bollinger_width_pct",
        "relative_volume", "volume_spike", "obv", "obv_rising", "support", "resistance",
        "distance_from_support_pct", "distance_from_resistance_pct", "adx", "di_plus",
        "di_minus", "volatility_regime",
    }


# --- known-input / known-output correctness ---


def test_rsi_matches_wilder_textbook_example():
    # Wilder's own worked example from "New Concepts in Technical Trading
    # Systems" — the canonical RSI-14 reference series/result.
    closes = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
        45.89, 46.03, 45.61, 46.28, 46.28,
    ]
    assert round(rsi(closes, period=14), 2) == 70.46


def test_ema_seeded_with_sma_then_smooths():
    # seed SMA of first 3 = 2.0, then one more value (8.0) smoothed in
    # with k=2/(3+1)=0.5 -> 8*0.5 + 2*0.5 = 5.0
    assert ema([2.0, 2.0, 2.0, 8.0], period=3) == 5.0


def test_bollinger_width_zero_on_constant_price():
    bb = bollinger_bands([100.0] * 25, period=20, num_std=2)
    assert bb["middle"] == 100.0
    assert bb["width_pct"] == 0.0


def test_bollinger_width_is_percent_not_absolute():
    # same absolute spread, different price scale -> different width_pct,
    # proving it's normalized rather than a raw currency-denominated width
    cheap = bollinger_bands([9.0, 10.0, 11.0] * 7, period=20, num_std=2)
    expensive = bollinger_bands([x * 1000 for x in [9.0, 10.0, 11.0] * 7], period=20, num_std=2)
    assert cheap["width_pct"] == pytest.approx(expensive["width_pct"])


def test_atr_constant_true_range():
    # every bar has high-low=2, no gaps -> ATR settles at 2.0
    highs = [11.0] * 20
    lows = [9.0] * 20
    closes = [10.0] * 20
    assert atr(highs, lows, closes, period=14) == 2.0


def test_relative_volume_double_average():
    volumes = [100.0] * 20 + [200.0]
    assert relative_volume(volumes, lookback=20) == 2.0


def test_volume_spike_true_above_threshold():
    assert volume_spike(3.0, threshold=2.0) is True
    assert volume_spike(1.5, threshold=2.0) is False
    assert volume_spike(None, threshold=2.0) is None


def test_obv_rising_on_uptrend():
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    volumes = [100.0] * 6
    result = obv(closes, volumes, slope_lookback=5)
    assert result["rising"] is True
    assert result["value"] == 500.0  # every bar up: +100 each, 5 up-moves


def test_obv_falling_on_downtrend():
    closes = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0]
    volumes = [100.0] * 6
    result = obv(closes, volumes, slope_lookback=5)
    assert result["rising"] is False


def test_support_resistance_min_max_over_lookback():
    highs = [10.0, 12.0, 9.0, 15.0, 11.0]
    lows = [8.0, 9.0, 7.0, 10.0, 9.5]
    result = support_resistance(highs, lows, lookback=5)
    assert result["support"] == 7.0
    assert result["resistance"] == 15.0


def test_volatility_bucket_boundaries():
    assert volatility_bucket(0.5, low_max=0.5, high_min=5.0) == "low"
    assert volatility_bucket(0.51, low_max=0.5, high_min=5.0) == "medium"
    assert volatility_bucket(4.99, low_max=0.5, high_min=5.0) == "medium"
    assert volatility_bucket(5.0, low_max=0.5, high_min=5.0) == "high"


def test_stoch_rsi_returns_k_and_d_within_bounds():
    # enough bars for rsi_period=14 + stoch_period=14 + smoothing
    closes = [44.0 + (i % 5) * 0.3 for i in range(40)]
    result = stoch_rsi(closes, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3)
    assert result is not None
    k, d = result
    assert 0.0 <= k <= 100.0
    assert 0.0 <= d <= 100.0


def test_adx_within_bounds_on_trending_series():
    n = 40
    highs = [10.0 + i * 0.5 for i in range(n)]
    lows = [9.0 + i * 0.5 for i in range(n)]
    closes = [9.5 + i * 0.5 for i in range(n)]
    result = adx(highs, lows, closes, period=14)
    assert result is not None
    assert 0.0 <= result["adx"] <= 100.0
    # a clean uptrend must show +DI dominating -DI
    assert result["di_plus"] > result["di_minus"]


def test_compute_features_sorts_descending_candles_ascending_first():
    # CoinDCX returns candles descending by time — feed them in that order
    # and confirm the engine still computes the RIGHT latest close (10, not
    # the first element in the list, 1).
    candles_desc = [_candle(i, i, i, i, i, 100) for i in range(10, 0, -1)]
    features = compute_features(candles_desc)
    assert features["close"] == 10


def test_compute_multi_timeframe_features_maps_each_timeframe():
    candles = [_candle(i, i, i, i, i, 100) for i in range(1, 30)]
    result = compute_multi_timeframe_features({"5m": candles, "1h": candles})
    assert set(result.keys()) == {"5m", "1h"}
    assert result["5m"]["close"] == 29
