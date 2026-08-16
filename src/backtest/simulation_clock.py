"""Drives the BacktestEngine's chronological replay. `is_bar_closed` is the
no-look-ahead rule's single source of truth — a bar is visible at
simulated time `t` only if its open-time + interval duration <= t (fully
closed, never the bar still forming at that instant). Confirmed empirically
against CoinDCX's live API before this was written: a no-range-filter
candle query always returns the still-forming bar as its most recent row,
so `time` is bar-OPEN time, not close time — this pins the rule exactly.

Also derives the simulation's own day boundary for daily-PnL bucketing —
deliberately does NOT import risk_manager.today_ist(), which is real
wall-clock time, not simulation-aware. TRADING_DAY_TZ itself (a bare
ZoneInfo constant, not a live clock call) is safe to reuse."""

from __future__ import annotations

from datetime import date, datetime, timezone

from src.agents.risk_manager import TRADING_DAY_TZ

TIMEFRAME_DURATION_MS: dict[str, int] = {
    "1m": 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def timeframe_duration_ms(interval: str) -> int:
    try:
        return TIMEFRAME_DURATION_MS[interval]
    except KeyError:
        raise ValueError(
            f"unsupported interval: {interval!r} (must be one of {sorted(TIMEFRAME_DURATION_MS)})"
        )


def is_bar_closed(bar_open_time_ms: int, interval: str, as_of_ms: int) -> bool:
    return bar_open_time_ms + timeframe_duration_ms(interval) <= as_of_ms


class SimulationClock:
    def __init__(self, start_ms: int, end_ms: int, tick_timeframe: str):
        if start_ms >= end_ms:
            raise ValueError("start_ms must be < end_ms")
        self._tick_ms = timeframe_duration_ms(tick_timeframe)
        self._start_ms = start_ms
        self._end_ms = end_ms
        self._now_ms = start_ms

    @property
    def now_ms(self) -> int:
        return self._now_ms

    @property
    def now(self) -> datetime:
        return datetime.fromtimestamp(self._now_ms / 1000, tz=timezone.utc)

    def today_ist(self) -> date:
        """Simulation-time equivalent of risk_manager.today_ist() — derived
        from clock state, never real wall-clock time."""
        return self.now.astimezone(TRADING_DAY_TZ).date()

    def ticks(self):
        """Yields each tick's epoch-ms timestamp in order, advancing `now`
        as it goes, up to and including end_ms."""
        t = self._start_ms
        while t <= self._end_ms:
            self._now_ms = t
            yield t
            t += self._tick_ms
