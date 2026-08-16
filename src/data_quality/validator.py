"""Validates raw CoinDCX candles before they reach the Feature Engine — no
candle skips this, live or backtest (see PROJECT_SPEC.md §3d). Pure
function, no DB/network access: takes a candle list, returns a
ValidationReport. What happens with a flagged issue (drop it, keep it,
drop the whole batch) is repair.py's job, driven by this report's severity
labels — this module only detects and classifies, never mutates.

Candle shape assumed throughout (CoinDCX's raw format, confirmed against
the live API): {open, high, low, close, volume, time} — time is bar-OPEN
time, epoch ms."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.config import (
    DATA_QUALITY_CLOCK_DRIFT_SECONDS_THRESHOLD,
    DATA_QUALITY_PRICE_SPIKE_PCT_THRESHOLD,
    DATA_QUALITY_SEVERITY_CLOCK_DRIFT,
    DATA_QUALITY_SEVERITY_DUPLICATE,
    DATA_QUALITY_SEVERITY_EXCHANGE_OUTAGE,
    DATA_QUALITY_SEVERITY_INVALID_OHLC,
    DATA_QUALITY_SEVERITY_MISSING_CANDLE,
    DATA_QUALITY_SEVERITY_NEGATIVE_PRICE,
    DATA_QUALITY_SEVERITY_OUT_OF_ORDER,
    DATA_QUALITY_SEVERITY_PRICE_SPIKE,
    DATA_QUALITY_SEVERITY_SYMBOL_MISMATCH,
    DATA_QUALITY_SEVERITY_TIMEFRAME_CHANGE,
    DATA_QUALITY_SEVERITY_TIMESTAMP_GAP,
    DATA_QUALITY_SEVERITY_ZERO_VOLUME,
)
SEVERITIES = ("ignore", "warn", "reject", "quarantine")

# Same mapping as src.backtest.simulation_clock.TIMEFRAME_DURATION_MS, kept
# as its own tiny copy rather than importing it — data_quality is used by
# live data_agent.py, and backtest already depends on live pure functions
# (feature engine, risk manager, statistics); importing backtest FROM a
# live-code path would invert that one-way layering for a 4-line dict.
_TIMEFRAME_DURATION_MS = {"1m": 60_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}


def timeframe_duration_ms(interval: str) -> int | None:
    """None for an interval outside CoinDCX's supported set — the
    validator degrades gracefully (skips duration-dependent checks) rather
    than crashing the whole fetch over an unexpected interval string."""
    return _TIMEFRAME_DURATION_MS.get(interval)


@dataclass
class Issue:
    issue_type: str
    severity: str
    detail: dict
    candle_time: int | None = None  # None = batch-level issue (e.g. exchange outage)


@dataclass
class ValidationReport:
    pair: str
    interval: str
    issues: list[Issue] = field(default_factory=list)
    quarantined: bool = False
    usable_candles: list[dict] = field(default_factory=list)

    def has_severity(self, severity: str) -> bool:
        return any(i.severity == severity for i in self.issues)


def _severity_for(check_name: str) -> str:
    return {
        "missing_candle": DATA_QUALITY_SEVERITY_MISSING_CANDLE,
        "duplicate": DATA_QUALITY_SEVERITY_DUPLICATE,
        "negative_price": DATA_QUALITY_SEVERITY_NEGATIVE_PRICE,
        "invalid_ohlc": DATA_QUALITY_SEVERITY_INVALID_OHLC,
        "out_of_order": DATA_QUALITY_SEVERITY_OUT_OF_ORDER,
        "timestamp_gap": DATA_QUALITY_SEVERITY_TIMESTAMP_GAP,
        "zero_volume": DATA_QUALITY_SEVERITY_ZERO_VOLUME,
        "price_spike": DATA_QUALITY_SEVERITY_PRICE_SPIKE,
        "exchange_outage": DATA_QUALITY_SEVERITY_EXCHANGE_OUTAGE,
        "clock_drift": DATA_QUALITY_SEVERITY_CLOCK_DRIFT,
        "symbol_mismatch": DATA_QUALITY_SEVERITY_SYMBOL_MISMATCH,
        "timeframe_change": DATA_QUALITY_SEVERITY_TIMEFRAME_CHANGE,
    }[check_name]


class MarketDataValidator:
    def validate(
        self,
        candles: list[dict],
        pair: str,
        interval: str,
        prior_candle: dict | None = None,
        expected_pair: str | None = None,
        live_fetch: bool = False,
    ) -> ValidationReport:
        """prior_candle: the last known-good candle before this batch (for
        gap/spike/order checks at the batch boundary), or None on a cold
        start. expected_pair: the pair this fetch was requested for — only
        checked against a candle's own 'pair'/'market' field when present
        (CoinDCX's candles response doesn't always echo it back). live_fetch:
        True only for data_agent.py's real-time path — clock drift is
        meaningless against historical/backtest data, which is never "now"."""
        report = ValidationReport(pair=pair, interval=interval)

        if not candles:
            report.issues.append(
                Issue("exchange_outage", _severity_for("exchange_outage"), {"reason": "empty response"})
            )
            if _severity_for("exchange_outage") in ("reject", "quarantine"):
                report.quarantined = True
            return report

        duration = timeframe_duration_ms(interval)
        seen_times: set[int] = set()
        prev = prior_candle

        for c in candles:
            t = c.get("time")

            if t in seen_times:
                report.issues.append(Issue("duplicate", _severity_for("duplicate"), {"time": t}, t))
            seen_times.add(t)

            for field_name in ("open", "high", "low", "close"):
                if c.get(field_name) is not None and c[field_name] <= 0:
                    report.issues.append(
                        Issue("negative_price", _severity_for("negative_price"), {"field": field_name, "value": c[field_name]}, t)
                    )

            o, h, l, cl = c.get("open"), c.get("high"), c.get("low"), c.get("close")
            if None not in (o, h, l, cl) and (h < l or o > h or o < l or cl > h or cl < l):
                report.issues.append(
                    Issue("invalid_ohlc", _severity_for("invalid_ohlc"), {"open": o, "high": h, "low": l, "close": cl}, t)
                )

            if c.get("volume") == 0:
                report.issues.append(Issue("zero_volume", _severity_for("zero_volume"), {}, t))

            if expected_pair is not None:
                candle_pair = c.get("pair") or c.get("market")
                if candle_pair is not None and candle_pair != expected_pair:
                    report.issues.append(
                        Issue(
                            "symbol_mismatch",
                            _severity_for("symbol_mismatch"),
                            {"expected": expected_pair, "got": candle_pair},
                            t,
                        )
                    )

            if prev is not None and t is not None and prev.get("time") is not None:
                delta = t - prev["time"]
                if delta < 0:
                    report.issues.append(
                        Issue("out_of_order", _severity_for("out_of_order"), {"prev_time": prev["time"], "time": t}, t)
                    )
                elif delta == 0:
                    pass  # already caught as duplicate above
                elif duration is not None and delta != duration:
                    if delta > duration:
                        bars_missing = int(delta / duration) - 1
                        report.issues.append(
                            Issue(
                                "missing_candle",
                                _severity_for("missing_candle"),
                                {"prev_time": prev["time"], "time": t, "bars_missing": bars_missing},
                                t,
                            )
                        )
                    else:
                        # delta > 0 but smaller than one bar's duration —
                        # the exchange returned a different bar spacing
                        # than requested (interval silently changed).
                        report.issues.append(
                            Issue(
                                "timeframe_change",
                                _severity_for("timeframe_change"),
                                {"expected_duration_ms": duration, "actual_delta_ms": delta},
                                t,
                            )
                        )

                if prev.get("close") and cl is not None and prev["close"] > 0:
                    pct_jump = abs(cl - prev["close"]) / prev["close"] * 100
                    if pct_jump >= DATA_QUALITY_PRICE_SPIKE_PCT_THRESHOLD:
                        report.issues.append(
                            Issue(
                                "price_spike",
                                _severity_for("price_spike"),
                                {"prev_close": prev["close"], "close": cl, "pct_jump": pct_jump},
                                t,
                            )
                        )

            prev = c

        if live_fetch and candles and duration is not None:
            last_open_ms = candles[-1].get("time")
            if last_open_ms is not None:
                implied_close_ms = last_open_ms + duration
                drift_seconds = (int(time.time() * 1000) - implied_close_ms) / 1000
                # Only flag drift the "wrong" direction — the still-forming
                # last bar being recent is expected, not drift.
                if drift_seconds > DATA_QUALITY_CLOCK_DRIFT_SECONDS_THRESHOLD:
                    report.issues.append(
                        Issue(
                            "clock_drift",
                            _severity_for("clock_drift"),
                            {"drift_seconds": drift_seconds},
                            last_open_ms,
                        )
                    )

        reject_times = {i.candle_time for i in report.issues if i.severity == "reject" and i.candle_time is not None}
        report.usable_candles = [c for c in candles if c.get("time") not in reject_times]

        if any(i.severity == "quarantine" for i in report.issues):
            report.quarantined = True
            report.usable_candles = []

        return report
