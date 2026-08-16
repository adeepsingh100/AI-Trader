from unittest.mock import patch

import pytest

from src.backtest.data_provider import CandleStore, fetch_historical_candles_paginated, ingest


def _candle(time_ms, o=100, h=101, l=99, c=100, v=10):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v, "time": time_ms}


# --- CandleStore: the no-look-ahead enforcement path ---


@patch("src.backtest.data_provider.models")
def test_visible_slice_excludes_still_forming_bar(mock_models):
    # Three 1m bars at t=0, 60000, 120000. Querying as_of=120000 (exactly
    # the open of the 3rd bar) must NOT return that bar — it's still
    # forming (closes at 180000) — but the bar at 60000 HAS closed by
    # 120000 (60000+60000=120000) and must be included.
    mock_models.get_historical_candles.return_value = [_candle(0), _candle(60_000), _candle(120_000)]
    store = CandleStore("I-BTC_INR", "1m", 0, 180_000)

    visible = store.visible_slice(as_of_ms=120_000, limit=10)

    assert [c["time"] for c in visible] == [0, 60_000]


@patch("src.backtest.data_provider.models")
def test_visible_slice_includes_bar_exactly_at_its_close_time(mock_models):
    mock_models.get_historical_candles.return_value = [_candle(0), _candle(60_000)]
    store = CandleStore("I-BTC_INR", "1m", 0, 120_000)

    visible = store.visible_slice(as_of_ms=60_000, limit=10)  # bar at t=0 closes at 60000

    assert [c["time"] for c in visible] == [0]


@patch("src.backtest.data_provider.models")
def test_visible_slice_respects_limit_taking_most_recent(mock_models):
    candles = [_candle(i * 60_000) for i in range(10)]
    mock_models.get_historical_candles.return_value = candles
    store = CandleStore("I-BTC_INR", "1m", 0, 600_000)

    visible = store.visible_slice(as_of_ms=9 * 60_000, limit=3)  # bars 0-8 closed, take last 3

    assert [c["time"] for c in visible] == [6 * 60_000, 7 * 60_000, 8 * 60_000]


@patch("src.backtest.data_provider.models")
def test_visible_slice_empty_before_any_bar_closes(mock_models):
    mock_models.get_historical_candles.return_value = [_candle(0)]
    store = CandleStore("I-BTC_INR", "1m", 0, 60_000)
    assert store.visible_slice(as_of_ms=0, limit=10) == []


@patch("src.backtest.data_provider.models")
def test_current_bar_open_price_is_the_still_forming_bar(mock_models):
    mock_models.get_historical_candles.return_value = [_candle(0, o=100), _candle(60_000, o=200)]
    store = CandleStore("I-BTC_INR", "1m", 0, 120_000)
    assert store.current_bar_open_price(as_of_ms=60_000) == 200
    assert store.current_bar_open_price(as_of_ms=90_000) == 200


@patch("src.backtest.data_provider.models")
def test_current_bar_open_price_none_before_series_starts(mock_models):
    mock_models.get_historical_candles.return_value = [_candle(60_000)]
    store = CandleStore("I-BTC_INR", "1m", 0, 120_000)
    assert store.current_bar_open_price(as_of_ms=0) is None


# --- pagination (network mocked, never real) ---


@patch("src.backtest.data_provider.BACKTEST_CANDLE_PAGE_SIZE", 2)
@patch("src.backtest.data_provider._fetch_page")
def test_fetch_historical_candles_paginated_walks_backward_until_start_covered(mock_fetch):
    # page cap = 2. First call returns the 2 most recent candles, second
    # call (cursor_end moved back) returns the 2 oldest, then it stops
    # because start_ms is covered.
    mock_fetch.side_effect = [
        [_candle(300_000), _candle(240_000)],
        [_candle(180_000), _candle(120_000)],
    ]

    candles = fetch_historical_candles_paginated("I-BTC_INR", "1m", start_ms=120_000, end_ms=300_000)

    assert [c["time"] for c in candles] == [120_000, 180_000, 240_000, 300_000]
    assert mock_fetch.call_count == 2


@patch("src.backtest.data_provider._fetch_page")
def test_fetch_historical_candles_paginated_stops_on_short_page(mock_fetch):
    mock_fetch.return_value = [_candle(0)]  # fewer than page size -> no more pages
    candles = fetch_historical_candles_paginated("I-BTC_INR", "1m", start_ms=0, end_ms=1_000_000)
    assert [c["time"] for c in candles] == [0]
    assert mock_fetch.call_count == 1


@patch("src.backtest.data_provider.models")
@patch("src.backtest.data_provider.fetch_historical_candles_paginated")
def test_ingest_upserts_fetched_candles(mock_fetch, mock_models):
    mock_fetch.return_value = [_candle(0), _candle(60_000)]
    count = ingest("I-BTC_INR", "1m", 0, 60_000)
    assert count == 2
    mock_models.upsert_historical_candles.assert_called_once_with("I-BTC_INR", "1m", mock_fetch.return_value)
