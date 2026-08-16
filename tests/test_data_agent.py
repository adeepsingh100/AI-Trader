from unittest.mock import patch

from src.agents.data_agent import get_market_snapshot


@patch("src.agents.data_agent.FEATURE_TIMEFRAMES", ["5m", "1h"])
@patch("src.agents.data_agent.FEATURE_CANDLE_LIMIT", 250)
@patch("src.agents.data_agent.get_candles")
@patch("src.agents.data_agent.symbol_to_pair")
@patch("src.agents.data_agent.top_inr_pairs_by_turnover")
@patch("src.agents.data_agent.get_markets_details")
def test_get_market_snapshot_fetches_one_call_per_symbol_per_timeframe(
    mock_details, mock_top, mock_pair, mock_candles
):
    mock_details.return_value = [{"symbol": "BTCINR", "pair": "I-BTC_INR"}]
    mock_top.return_value = [{"market": "BTCINR", "last_price": 1_000_000, "turnover_inr": 5}]
    mock_pair.return_value = "I-BTC_INR"
    mock_candles.return_value = [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "time": 1}]

    snapshot = get_market_snapshot(n=1)

    assert len(snapshot) == 1
    assert mock_candles.call_count == 2
    mock_candles.assert_any_call("I-BTC_INR", interval="5m", limit=250)
    mock_candles.assert_any_call("I-BTC_INR", interval="1h", limit=250)


@patch("src.agents.data_agent.FEATURE_TIMEFRAMES", ["5m"])
@patch("src.agents.data_agent.get_candles")
@patch("src.agents.data_agent.symbol_to_pair")
@patch("src.agents.data_agent.top_inr_pairs_by_turnover")
@patch("src.agents.data_agent.get_markets_details")
def test_get_market_snapshot_shape_has_no_orderbook_key(
    mock_details, mock_top, mock_pair, mock_candles
):
    mock_details.return_value = []
    mock_top.return_value = [{"market": "ETHINR", "last_price": 200_000, "turnover_inr": 3}]
    mock_pair.return_value = "I-ETH_INR"
    mock_candles.return_value = []

    snapshot = get_market_snapshot(n=1)

    market = snapshot[0]
    assert "orderbook" not in market
    assert market["candles_by_timeframe"] == {"5m": []}
    assert market["symbol"] == "ETHINR"
    assert market["pair"] == "I-ETH_INR"
    assert market["last_price"] == 200_000
