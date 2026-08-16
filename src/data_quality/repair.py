"""Auto-repairs what validator.py flagged as safely fixable. Never touches
a reject/quarantine-severity issue — those candles are already excluded
from ValidationReport.usable_candles and stay excluded; repair only
resolves warn-tier issues with a defined fix. Every repair returns a
logged entry — nothing here silently mutates data (see PROJECT_SPEC.md
§3d and the user's explicit "never silently modify data" instruction)."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import DATA_REPAIR_MAX_GAP_BARS
from src.data_quality.validator import ValidationReport, timeframe_duration_ms

_REPAIRABLE_TYPES = {"duplicate", "out_of_order", "missing_candle"}


@dataclass
class RepairLogEntry:
    repair_type: str
    detail: dict
    candle_time: int | None = None


def _dedup_keep_latest(candles: list[dict]) -> list[dict]:
    """Exact-duplicate-timestamp merge: keeps the last-seen row for a given
    time (the most recently fetched value for that bar), matching how a
    re-fetch of the same window would naturally supersede an earlier one."""
    by_time: dict[int, dict] = {}
    for c in candles:
        by_time[c["time"]] = c
    return [by_time[t] for t in by_time]


def _interpolate_gap(before: dict, after: dict, missing_times: list[int]) -> list[dict]:
    """Linear interpolation between the two known-good bars bracketing the
    gap — open/high/low/close all move linearly from `before`'s close to
    `after`'s open, volume is the average of the two neighbors (no better
    signal exists for a bar that never actually printed)."""
    span = after["time"] - before["time"]
    start_price, end_price = before["close"], after["open"]
    filled = []
    for t in missing_times:
        frac = (t - before["time"]) / span
        price = start_price + (end_price - start_price) * frac
        filled.append(
            {
                "time": t,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": (before["volume"] + after["volume"]) / 2,
            }
        )
    return filled


class DataRepairEngine:
    def repair(
        self, candles: list[dict], report: ValidationReport, pair: str, interval: str
    ) -> tuple[list[dict], list[RepairLogEntry]]:
        if report.quarantined:
            return [], []

        log: list[RepairLogEntry] = []
        working = list(candles)

        if report.has_severity("warn") and any(i.issue_type == "duplicate" for i in report.issues):
            before = len(working)
            working = _dedup_keep_latest(working)
            if len(working) != before:
                log.append(RepairLogEntry("duplicate_merge", {"removed": before - len(working)}))

        working.sort(key=lambda c: c["time"])
        if any(i.issue_type == "out_of_order" for i in report.issues):
            log.append(RepairLogEntry("reorder", {"count": len(working)}))

        duration = timeframe_duration_ms(interval)
        gap_issues = [
            i
            for i in report.issues
            if i.issue_type == "missing_candle" and i.detail.get("bars_missing", 0) <= DATA_REPAIR_MAX_GAP_BARS
        ]
        for issue in gap_issues:
            prev_time, cur_time = issue.detail["prev_time"], issue.detail["time"]
            before_candle = next((c for c in working if c["time"] == prev_time), None)
            after_candle = next((c for c in working if c["time"] == cur_time), None)
            if before_candle is None or after_candle is None:
                continue
            missing_times = list(range(prev_time + duration, cur_time, duration))
            filled = _interpolate_gap(before_candle, after_candle, missing_times)
            working.extend(filled)
            log.append(
                RepairLogEntry(
                    "gap_interpolation",
                    {"prev_time": prev_time, "time": cur_time, "bars_filled": len(filled)},
                    cur_time,
                )
            )

        working.sort(key=lambda c: c["time"])
        return working, log
