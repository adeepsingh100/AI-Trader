"""Historical OHLCV: paginated network fetch (CoinDCX's public candles
endpoint supports startTime/endTime even though src/coindcx_client.py's
live get_candles() wrapper never exposes them — confirmed empirically) and
an in-memory CandleStore that serves closed-bar-only slices during a
backtest run's hot loop. Ingestion (network + historical_candles cache
writes) happens once, up front, via ingest_data.py — the hot loop only
ever reads from an already-loaded CandleStore, zero network/DB calls."""

from __future__ import annotations

import bisect

import requests

from src.backtest.simulation_clock import timeframe_duration_ms
from src.config import BACKTEST_CANDLE_PAGE_SIZE
from src.db import models

CANDLES_URL = "https://public.coindcx.com/market_data/candles"


def _fetch_page(pair: str, interval: str, start_ms: int, end_ms: int, limit: int) -> list[dict]:
    resp = requests.get(
        CANDLES_URL,
        params={"pair": pair, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_historical_candles_paginated(pair: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """Walks backward from end_ms in BACKTEST_CANDLE_PAGE_SIZE-candle pages
    (the exchange's own per-call cap, confirmed empirically — each call
    returns at most that many candles regardless of range width) until
    start_ms is covered. Returns ascending by time, deduped by time."""
    seen: dict[int, dict] = {}
    cursor_end = end_ms
    while True:
        page = _fetch_page(pair, interval, start_ms, cursor_end, BACKTEST_CANDLE_PAGE_SIZE)
        if not page:
            break
        for c in page:
            seen[c["time"]] = c
        earliest = min(c["time"] for c in page)
        if earliest <= start_ms or len(page) < BACKTEST_CANDLE_PAGE_SIZE:
            break
        cursor_end = earliest - 1
    return sorted(seen.values(), key=lambda c: c["time"])


def ingest(pair: str, interval: str, start_ms: int, end_ms: int) -> int:
    """Fetches + upserts into the historical_candles cache. Returns the
    number of candles ingested. Only ever called by ingest_data.py, never
    from inside a backtest run itself."""
    candles = fetch_historical_candles_paginated(pair, interval, start_ms, end_ms)
    models.upsert_historical_candles(pair, interval, candles)
    return len(candles)


class CandleStore:
    """Loaded once (from the historical_candles cache, never live network)
    at the start of a run, read-only afterward. One instance per
    (pair, interval). visible_slice() is the only way engine code reads
    bars — closed-bar-only is structural here, not a discipline to
    remember at each call site."""

    def __init__(self, pair: str, interval: str, start_ms: int, end_ms: int):
        self.pair = pair
        self.interval = interval
        self._duration = timeframe_duration_ms(interval)
        rows = models.get_historical_candles(pair, interval, start_ms, end_ms)
        rows = sorted(rows, key=lambda c: c["time"])
        self._times = [r["time"] for r in rows]
        self._rows = rows

    def visible_slice(self, as_of_ms: int, limit: int) -> list[dict]:
        """Up to `limit` most-recent candles fully closed as of `as_of_ms`.
        Cutoff = as_of_ms - duration is the bisect-friendly rearrangement
        of simulation_clock.is_bar_closed's per-row rule (open_time +
        duration <= as_of_ms <=> open_time <= as_of_ms - duration) — same
        rule, vectorized for a hot-loop-safe O(log n) lookup instead of an
        O(n) per-tick scan."""
        cutoff = as_of_ms - self._duration
        idx = bisect.bisect_right(self._times, cutoff)
        if idx == 0:
            return []
        start = max(0, idx - limit)
        return self._rows[start:idx]

    def current_bar_open_price(self, as_of_ms: int) -> float | None:
        """Ticker-price proxy for `now` (mirrors live's real-time
        get_ticker() last_price) — the open of the bar currently forming
        at as_of_ms, i.e. the most recent bar whose open_time <= as_of_ms.
        Deliberately NOT closed-bar-filtered (that's visible_slice's job
        for feature computation) — a real-time price proxy IS supposed to
        reflect the in-progress bar, same as a live ticker would."""
        idx = bisect.bisect_right(self._times, as_of_ms) - 1
        if idx < 0:
            return None
        return float(self._rows[idx]["open"])
