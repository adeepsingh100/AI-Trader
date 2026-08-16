from datetime import date
from unittest.mock import patch

from src.backtest.ingest_data import estimate_row_count, ingest_universe, resolve_symbol_to_pair


@patch("src.backtest.ingest_data.symbol_to_pair")
@patch("src.backtest.ingest_data.get_markets_details")
def test_resolve_symbol_to_pair_maps_every_symbol(mock_details, mock_symbol_to_pair):
    mock_details.return_value = [{"symbol": "BTCINR", "pair": "I-BTC_INR"}]
    mock_symbol_to_pair.side_effect = lambda s, details: {"BTCINR": "I-BTC_INR", "ETHINR": "I-ETH_INR"}[s]

    result = resolve_symbol_to_pair(["BTCINR", "ETHINR"])

    assert result == {"BTCINR": "I-BTC_INR", "ETHINR": "I-ETH_INR"}
    mock_details.assert_called_once()  # fetched once, not once per symbol


def test_estimate_row_count_sums_all_four_timeframes():
    # 1 day, 1 symbol: 1440 (1m) + 96 (15m) + 24 (1h) + 1 (1d) = 1561
    assert estimate_row_count(n_symbols=1, n_days=1) == 1440 + 96 + 24 + 1


def test_estimate_row_count_scales_with_symbol_count():
    assert estimate_row_count(n_symbols=3, n_days=1) == 3 * estimate_row_count(n_symbols=1, n_days=1)


@patch("src.backtest.ingest_data.ingest")
@patch("src.backtest.ingest_data.resolve_symbol_to_pair")
def test_ingest_universe_ingests_every_symbol_timeframe_pair(mock_resolve, mock_ingest):
    mock_resolve.return_value = {"BTCINR": "I-BTC_INR"}
    mock_ingest.return_value = 100

    counts = ingest_universe(["BTCINR"], date(2024, 1, 1), date(2024, 1, 2), warmup_buffer_days=0)

    assert set(counts.keys()) == {"BTCINR:1m", "BTCINR:15m", "BTCINR:1h", "BTCINR:1d"}
    assert all(v == 100 for v in counts.values())
    assert mock_ingest.call_count == 4


@patch("src.backtest.ingest_data.ingest")
@patch("src.backtest.ingest_data.resolve_symbol_to_pair")
def test_ingest_universe_widens_range_by_warmup_buffer(mock_resolve, mock_ingest):
    from src.backtest.ingest_data import _date_to_ms

    mock_resolve.return_value = {"BTCINR": "I-BTC_INR"}
    mock_ingest.return_value = 0

    ingest_universe(["BTCINR"], date(2024, 1, 10), date(2024, 1, 11), warmup_buffer_days=5)

    call = mock_ingest.call_args_list[0]
    pair, interval, start_ms, end_ms = call.args
    assert start_ms == _date_to_ms(date(2024, 1, 5))
