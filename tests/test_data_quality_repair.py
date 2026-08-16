from src.data_quality.repair import DataRepairEngine
from src.data_quality.validator import MarketDataValidator


def _candle(t, o=100, h=101, l=99, c=100.5, v=1000):
    return {"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_repair_clean_candles_returns_unchanged_with_no_log():
    validator, repairer = MarketDataValidator(), DataRepairEngine()
    candles = [_candle(0), _candle(60_000)]
    report = validator.validate(candles, "I-BTC_INR", "1m")
    repaired, log = repairer.repair(report.usable_candles, report, "I-BTC_INR", "1m")
    assert repaired == candles
    assert log == []


def test_repair_merges_exact_duplicates_keeping_latest():
    validator, repairer = MarketDataValidator(), DataRepairEngine()
    older = _candle(0, o=100, h=101, l=99, c=100.5)
    newer = _candle(0, o=100, h=101, l=99, c=100.8)  # same time, re-fetched with a slightly different close
    report = validator.validate([older, newer], "I-BTC_INR", "1m")
    repaired, log = repairer.repair(report.usable_candles, report, "I-BTC_INR", "1m")
    assert len(repaired) == 1
    assert repaired[0]["close"] == 100.8
    assert any(entry.repair_type == "duplicate_merge" for entry in log)


def test_repair_interpolates_small_gap():
    validator, repairer = MarketDataValidator(), DataRepairEngine()
    before = _candle(0, o=100, h=101, l=99, c=100)
    after = _candle(180_000, o=110, h=111, l=109, c=110)  # 2 bars missing (within DATA_REPAIR_MAX_GAP_BARS default 3)
    report = validator.validate([before, after], "I-BTC_INR", "1m")
    repaired, log = repairer.repair(report.usable_candles, report, "I-BTC_INR", "1m")
    times = sorted(c["time"] for c in repaired)
    assert times == [0, 60_000, 120_000, 180_000]
    assert any(entry.repair_type == "gap_interpolation" for entry in log)
    # interpolated prices move linearly from before.close to after.open
    filled_60k = next(c for c in repaired if c["time"] == 60_000)
    assert 100 < filled_60k["close"] < 110


def test_repair_does_not_fill_a_gap_wider_than_max_gap_bars(monkeypatch):
    monkeypatch.setattr("src.data_quality.repair.DATA_REPAIR_MAX_GAP_BARS", 1)
    validator, repairer = MarketDataValidator(), DataRepairEngine()
    before = _candle(0)
    after = _candle(300_000)  # 4 bars missing, exceeds the max
    report = validator.validate([before, after], "I-BTC_INR", "1m")
    repaired, log = repairer.repair(report.usable_candles, report, "I-BTC_INR", "1m")
    assert len(repaired) == 2  # unchanged — gap left unrepaired
    assert not any(entry.repair_type == "gap_interpolation" for entry in log)


def test_repair_never_touches_quarantined_report():
    validator, repairer = MarketDataValidator(), DataRepairEngine()
    report = validator.validate([], "I-BTC_INR", "1m")  # exchange_outage -> reject by default, not quarantine
    report.quarantined = True  # force the quarantine branch explicitly
    repaired, log = repairer.repair([_candle(0)], report, "I-BTC_INR", "1m")
    assert repaired == []
    assert log == []


def test_repair_reorders_out_of_order_candles():
    validator, repairer = MarketDataValidator(), DataRepairEngine()
    candles = [_candle(60_000), _candle(0)]
    report = validator.validate(candles, "I-BTC_INR", "1m")
    repaired, log = repairer.repair(report.usable_candles, report, "I-BTC_INR", "1m")
    assert [c["time"] for c in repaired] == [0, 60_000]
    assert any(entry.repair_type == "reorder" for entry in log)
