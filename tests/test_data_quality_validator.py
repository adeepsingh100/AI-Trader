from unittest.mock import patch

from src.data_quality.validator import MarketDataValidator


def _candle(t, o=100, h=101, l=99, c=100.5, v=1000):
    return {"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_validate_clean_candles_has_no_issues():
    validator = MarketDataValidator()
    candles = [_candle(0), _candle(60_000), _candle(120_000)]
    report = validator.validate(candles, "I-BTC_INR", "1m")
    assert report.issues == []
    assert report.usable_candles == candles


def test_validate_empty_response_flags_exchange_outage():
    validator = MarketDataValidator()
    report = validator.validate([], "I-BTC_INR", "1m")
    assert any(i.issue_type == "exchange_outage" for i in report.issues)


def test_validate_negative_price_rejected_by_default():
    validator = MarketDataValidator()
    candles = [_candle(0, o=-5)]
    report = validator.validate(candles, "I-BTC_INR", "1m")
    assert any(i.issue_type == "negative_price" and i.severity == "reject" for i in report.issues)
    assert report.usable_candles == []


def test_validate_invalid_ohlc_high_below_low():
    validator = MarketDataValidator()
    candles = [_candle(0, h=90, l=99)]
    report = validator.validate(candles, "I-BTC_INR", "1m")
    assert any(i.issue_type == "invalid_ohlc" for i in report.issues)


def test_validate_duplicate_timestamp_flagged():
    validator = MarketDataValidator()
    candles = [_candle(0), _candle(0)]
    report = validator.validate(candles, "I-BTC_INR", "1m")
    assert any(i.issue_type == "duplicate" for i in report.issues)


def test_validate_out_of_order_timestamp_flagged():
    validator = MarketDataValidator()
    candles = [_candle(60_000), _candle(0)]
    report = validator.validate(candles, "I-BTC_INR", "1m")
    assert any(i.issue_type == "out_of_order" for i in report.issues)


def test_validate_missing_candle_gap_flagged():
    validator = MarketDataValidator()
    candles = [_candle(0), _candle(180_000)]  # 2 bars missing (60_000, 120_000)
    report = validator.validate(candles, "I-BTC_INR", "1m")
    missing = [i for i in report.issues if i.issue_type == "missing_candle"]
    assert missing and missing[0].detail["bars_missing"] == 2


def test_validate_zero_volume_flagged():
    validator = MarketDataValidator()
    candles = [_candle(0, v=0)]
    report = validator.validate(candles, "I-BTC_INR", "1m")
    assert any(i.issue_type == "zero_volume" for i in report.issues)


def test_validate_price_spike_flagged():
    validator = MarketDataValidator()
    candles = [_candle(0, c=100), _candle(60_000, o=1000, c=1000)]  # 900% jump
    report = validator.validate(candles, "I-BTC_INR", "1m")
    assert any(i.issue_type == "price_spike" for i in report.issues)


def test_validate_symbol_mismatch_flagged_when_pair_field_present():
    validator = MarketDataValidator()
    candles = [{**_candle(0), "pair": "I-ETH_INR"}]
    report = validator.validate(candles, "I-BTC_INR", "1m", expected_pair="I-BTC_INR")
    assert any(i.issue_type == "symbol_mismatch" for i in report.issues)


def test_validate_unknown_interval_degrades_gracefully_no_crash():
    """A synthetic/unexpected interval string (outside CoinDCX's real
    {1m,15m,1h,1d} set) must not crash the validator — duration-dependent
    checks (missing_candle/timeframe_change/clock_drift) are simply
    skipped rather than raising."""
    validator = MarketDataValidator()
    candles = [_candle(0), _candle(300_000)]
    report = validator.validate(candles, "I-BTC_INR", "5m", live_fetch=True)
    assert not any(i.issue_type in ("missing_candle", "timeframe_change", "clock_drift") for i in report.issues)


def test_validate_clock_drift_flagged_on_stale_live_fetch():
    validator = MarketDataValidator()
    stale_time_ms = 0  # epoch start — wildly stale vs. real "now"
    candles = [_candle(stale_time_ms)]
    with patch("src.data_quality.validator.time.time", return_value=1_700_000_000):
        report = validator.validate(candles, "I-BTC_INR", "1m", live_fetch=True)
    assert any(i.issue_type == "clock_drift" for i in report.issues)


def test_validate_clock_drift_not_checked_for_backtest_ingest():
    validator = MarketDataValidator()
    candles = [_candle(0)]
    report = validator.validate(candles, "I-BTC_INR", "1m", live_fetch=False)
    assert not any(i.issue_type == "clock_drift" for i in report.issues)


def test_validate_quarantine_severity_empties_usable_candles(monkeypatch):
    monkeypatch.setattr("src.data_quality.validator.DATA_QUALITY_SEVERITY_NEGATIVE_PRICE", "quarantine")
    validator = MarketDataValidator()
    candles = [_candle(0, o=-1), _candle(60_000)]
    report = validator.validate(candles, "I-BTC_INR", "1m")
    assert report.quarantined is True
    assert report.usable_candles == []
