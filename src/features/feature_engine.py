"""Technical indicators, computed from candles alone. Every public function
returns None on insufficient history instead of raising — a short series
(new listing, thin timeframe) must degrade a feature to "unknown", never
crash the scoring pass built on top of this module. Never makes a trading
decision — that's OpportunityScorer's job (src/features/opportunity_scorer.py).

Candle dicts are CoinDCX's shape: {open, high, low, close, volume, time},
returned DESCENDING by time. Every candle-consuming function here sorts
ascending by `time` first rather than trusting the caller's ordering."""

from __future__ import annotations

from src.config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BOLLINGER_NUM_STD,
    BOLLINGER_PERIOD,
    EMA_TREND_PERIOD_1,
    EMA_TREND_PERIOD_2,
    EMA_TREND_PERIOD_3,
    EMA_TREND_PERIOD_4,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    OBV_SLOPE_LOOKBACK,
    RELATIVE_VOLUME_LOOKBACK,
    RSI_PERIOD,
    STOCH_D_SMOOTH,
    STOCH_K_SMOOTH,
    STOCH_RSI_PERIOD,
    SUPPORT_RESISTANCE_LOOKBACK,
    VOLATILITY_HIGH_MIN_PCT,
    VOLATILITY_LOW_MAX_PCT,
    VOLUME_SPIKE_THRESHOLD,
)

FEATURE_KEYS = [
    "close",
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_200",
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "atr",
    "atr_pct",
    "bollinger_width_pct",
    "relative_volume",
    "volume_spike",
    "obv",
    "obv_rising",
    "support",
    "resistance",
    "distance_from_support_pct",
    "distance_from_resistance_pct",
    "adx",
    "di_plus",
    "di_minus",
    "volatility_regime",
]


# --- private series helpers (return a value per bar, not just the latest) --


def _ema_series(values: list[float], period: int) -> list[float]:
    if period <= 0 or len(values) < period:
        return []
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    series = [e]
    for v in values[period:]:
        e = v * k + e * (1 - k)
        series.append(e)
    return series


def _sma_series(values: list[float], period: int) -> list[float]:
    if period <= 0 or len(values) < period:
        return []
    return [sum(values[i - period + 1 : i + 1]) / period for i in range(period - 1, len(values))]


def _wilder_smooth_series(values: list[float], period: int) -> list[float]:
    if period <= 0 or len(values) < period:
        return []
    avg = sum(values[:period]) / period
    series = [avg]
    for v in values[period:]:
        avg = (avg * (period - 1) + v) / period
        series.append(avg)
    return series


def _rsi_series(closes: list[float], period: int) -> list[float]:
    if period <= 0 or len(closes) < period + 1:
        return []
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    def _value(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    series = [_value(avg_gain, avg_loss)]
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        series.append(_value(avg_gain, avg_loss))
    return series


# --- public indicators (latest value only) ----------------------------------


def ema(values: list[float], period: int) -> float | None:
    series = _ema_series(values, period)
    return series[-1] if series else None


def rsi(closes: list[float], period: int = RSI_PERIOD) -> float | None:
    series = _rsi_series(closes, period)
    return series[-1] if series else None


def macd(
    closes: list[float],
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[float, float, float] | None:
    if len(closes) < slow:
        return None
    fast_series = _ema_series(closes, fast)
    slow_series = _ema_series(closes, slow)
    if not fast_series or not slow_series:
        return None
    # Both series end at the same last close — align on the tail, since
    # fast_series (shorter period, longer series) starts earlier in time.
    macd_series = [f - s for f, s in zip(fast_series[-len(slow_series) :], slow_series)]
    if len(macd_series) < signal:
        return None
    signal_series = _ema_series(macd_series, signal)
    if not signal_series:
        return None
    macd_val = macd_series[-1]
    signal_val = signal_series[-1]
    return macd_val, signal_val, macd_val - signal_val


def stoch_rsi(
    closes: list[float],
    rsi_period: int = STOCH_RSI_PERIOD,
    stoch_period: int = STOCH_RSI_PERIOD,
    k_smooth: int = STOCH_K_SMOOTH,
    d_smooth: int = STOCH_D_SMOOTH,
) -> tuple[float, float] | None:
    rsi_series = _rsi_series(closes, rsi_period)
    if len(rsi_series) < stoch_period:
        return None
    stoch_values = []
    for i in range(stoch_period - 1, len(rsi_series)):
        window = rsi_series[i - stoch_period + 1 : i + 1]
        lo, hi = min(window), max(window)
        stoch_values.append(0.0 if hi == lo else (rsi_series[i] - lo) / (hi - lo) * 100)
    k_series = _sma_series(stoch_values, k_smooth)
    if not k_series:
        return None
    d_series = _sma_series(k_series, d_smooth)
    if not d_series:
        return None
    return k_series[-1], d_series[-1]


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = ATR_PERIOD) -> float | None:
    if len(closes) < period + 1:
        return None
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    series = _wilder_smooth_series(trs, period)
    return series[-1] if series else None


def bollinger_bands(
    closes: list[float], period: int = BOLLINGER_PERIOD, num_std: float = BOLLINGER_NUM_STD
) -> dict | None:
    if len(closes) < period:
        return None
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((c - middle) ** 2 for c in window) / period
    std = variance**0.5
    upper = middle + num_std * std
    lower = middle - num_std * std
    width_pct = (upper - lower) / middle * 100 if middle else None
    return {"upper": upper, "middle": middle, "lower": lower, "width_pct": width_pct}


def relative_volume(volumes: list[float], lookback: int = RELATIVE_VOLUME_LOOKBACK) -> float | None:
    if len(volumes) < lookback + 1:
        return None
    avg = sum(volumes[-lookback - 1 : -1]) / lookback
    if avg == 0:
        return None
    return volumes[-1] / avg


def volume_spike(relative_vol: float | None, threshold: float = VOLUME_SPIKE_THRESHOLD) -> bool | None:
    if relative_vol is None:
        return None
    return relative_vol >= threshold


def obv(closes: list[float], volumes: list[float], slope_lookback: int = OBV_SLOPE_LOOKBACK) -> dict | None:
    if len(closes) < 2 or len(closes) != len(volumes):
        return None
    series = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            series.append(series[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            series.append(series[-1] - volumes[i])
        else:
            series.append(series[-1])
    if len(series) < slope_lookback + 1:
        return None
    return {"value": series[-1], "rising": series[-1] > series[-1 - slope_lookback]}


def support_resistance(
    highs: list[float], lows: list[float], lookback: int = SUPPORT_RESISTANCE_LOOKBACK
) -> dict | None:
    if len(highs) < 2 or len(lows) < 2:
        return None
    return {"support": min(lows[-lookback:]), "resistance": max(highs[-lookback:])}


def adx(highs: list[float], lows: list[float], closes: list[float], period: int = ADX_PERIOD) -> dict | None:
    n = len(closes)
    if n < period * 2 + 1:
        return None
    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))

    smoothed_tr = _wilder_smooth_series(trs, period)
    smoothed_plus_dm = _wilder_smooth_series(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth_series(minus_dm, period)
    if not smoothed_tr or not smoothed_plus_dm or not smoothed_minus_dm:
        return None

    dx_series = []
    for tr, pdm, mdm in zip(smoothed_tr, smoothed_plus_dm, smoothed_minus_dm):
        if tr == 0:
            dx_series.append(0.0)
            continue
        plus_di = 100 * pdm / tr
        minus_di = 100 * mdm / tr
        denom = plus_di + minus_di
        dx_series.append(0.0 if denom == 0 else 100 * abs(plus_di - minus_di) / denom)

    adx_series = _wilder_smooth_series(dx_series, period)
    if not adx_series:
        return None

    final_tr, final_pdm, final_mdm = smoothed_tr[-1], smoothed_plus_dm[-1], smoothed_minus_dm[-1]
    final_plus_di = 0.0 if final_tr == 0 else 100 * final_pdm / final_tr
    final_minus_di = 0.0 if final_tr == 0 else 100 * final_mdm / final_tr
    return {"adx": adx_series[-1], "di_plus": final_plus_di, "di_minus": final_minus_di}


def volatility_bucket(
    atr_pct: float, low_max: float = VOLATILITY_LOW_MAX_PCT, high_min: float = VOLATILITY_HIGH_MIN_PCT
) -> str:
    if atr_pct <= low_max:
        return "low"
    if atr_pct >= high_min:
        return "high"
    return "medium"


# --- entry points -------------------------------------------------------


def compute_features(candles: list[dict]) -> dict:
    features = dict.fromkeys(FEATURE_KEYS, None)
    if not candles:
        return features

    ordered = sorted(candles, key=lambda c: c["time"])
    closes = [float(c["close"]) for c in ordered]
    highs = [float(c["high"]) for c in ordered]
    lows = [float(c["low"]) for c in ordered]
    volumes = [float(c["volume"]) for c in ordered]
    close = closes[-1]
    features["close"] = close

    features["ema_20"] = ema(closes, EMA_TREND_PERIOD_1)
    features["ema_50"] = ema(closes, EMA_TREND_PERIOD_2)
    features["ema_100"] = ema(closes, EMA_TREND_PERIOD_3)
    features["ema_200"] = ema(closes, EMA_TREND_PERIOD_4)

    features["rsi"] = rsi(closes)

    macd_result = macd(closes)
    if macd_result is not None:
        features["macd_line"], features["macd_signal"], features["macd_histogram"] = macd_result

    stoch_result = stoch_rsi(closes)
    if stoch_result is not None:
        features["stoch_rsi_k"], features["stoch_rsi_d"] = stoch_result

    atr_val = atr(highs, lows, closes)
    features["atr"] = atr_val
    if atr_val is not None and close:
        features["atr_pct"] = atr_val / close * 100

    bb = bollinger_bands(closes)
    if bb is not None:
        features["bollinger_width_pct"] = bb["width_pct"]

    rel_vol = relative_volume(volumes)
    features["relative_volume"] = rel_vol
    features["volume_spike"] = volume_spike(rel_vol)

    obv_result = obv(closes, volumes)
    if obv_result is not None:
        features["obv"] = obv_result["value"]
        features["obv_rising"] = obv_result["rising"]

    sr = support_resistance(highs, lows)
    if sr is not None:
        features["support"] = sr["support"]
        features["resistance"] = sr["resistance"]
        if close:
            features["distance_from_support_pct"] = (close - sr["support"]) / close * 100
            features["distance_from_resistance_pct"] = (sr["resistance"] - close) / close * 100

    adx_result = adx(highs, lows, closes)
    if adx_result is not None:
        features["adx"] = adx_result["adx"]
        features["di_plus"] = adx_result["di_plus"]
        features["di_minus"] = adx_result["di_minus"]

    if features["atr_pct"] is not None:
        features["volatility_regime"] = volatility_bucket(features["atr_pct"])

    return features


def compute_multi_timeframe_features(candles_by_timeframe: dict[str, list[dict]]) -> dict[str, dict]:
    return {tf: compute_features(candles) for tf, candles in candles_by_timeframe.items()}
