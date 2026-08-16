from unittest.mock import Mock, patch

import requests

from src.coindcx_client import (
    create_order,
    get_balances,
    get_candles,
    get_order_status,
    get_ticker,
    top_inr_pairs_by_turnover,
)


def test_top_inr_pairs_by_turnover_filters_and_ranks():
    ticker = [
        {"market": "BTCINR", "volume": "10", "last_price": "100"},   # turnover 1000
        {"market": "ETHINR", "volume": "5", "last_price": "500"},    # turnover 2500
        {"market": "BTCUSDT", "volume": "999", "last_price": "999"}, # not INR, excluded
        {"market": "XRPINR", "volume": "1", "last_price": "1"},      # turnover 1
    ]

    top = top_inr_pairs_by_turnover(n=2, ticker=ticker)

    assert [t["market"] for t in top] == ["ETHINR", "BTCINR"]
    assert top[0]["turnover_inr"] == 2500.0


# --- resilience: reads retry, writes (create_order) never do ---


@patch("src.coindcx_client.requests.get")
def test_get_ticker_retries_on_transient_failure(mock_get):
    ok_response = Mock(json=Mock(return_value=[{"market": "BTCINR"}]), raise_for_status=Mock())
    mock_get.side_effect = [requests.ConnectionError("blip"), ok_response]

    with patch("src.resilience.time.sleep"):
        result = get_ticker()

    assert result == [{"market": "BTCINR"}]
    assert mock_get.call_count == 2


@patch("src.coindcx_client.requests.get")
def test_get_candles_retries_on_transient_failure(mock_get):
    ok_response = Mock(json=Mock(return_value=[{"time": 1}]), raise_for_status=Mock())
    mock_get.side_effect = [requests.ConnectionError("blip"), ok_response]

    with patch("src.resilience.time.sleep"):
        result = get_candles("I-BTC_INR")

    assert result == [{"time": 1}]
    assert mock_get.call_count == 2


@patch("src.coindcx_client.requests.get")
def test_get_ticker_raises_after_exhausting_retries(mock_get):
    mock_get.side_effect = requests.ConnectionError("down")

    with patch("src.resilience.time.sleep"):
        try:
            get_ticker()
            assert False, "expected ConnectionError"
        except requests.ConnectionError:
            pass

    from src.config import RETRY_MAX_ATTEMPTS

    assert mock_get.call_count == RETRY_MAX_ATTEMPTS


@patch("src.coindcx_client._signed_post")
def test_get_balances_retries_on_transient_failure(mock_post):
    mock_post.side_effect = [requests.ConnectionError("blip"), [{"currency": "INR", "balance": "10"}]]

    with patch("src.resilience.time.sleep"):
        result = get_balances()

    assert result == [{"currency": "INR", "balance": "10"}]
    assert mock_post.call_count == 2


@patch("src.coindcx_client._signed_post")
def test_get_order_status_retries_on_transient_failure(mock_post):
    mock_post.side_effect = [requests.ConnectionError("blip"), {"status": "filled"}]

    with patch("src.resilience.time.sleep"):
        result = get_order_status("order-1")

    assert result == {"status": "filled"}
    assert mock_post.call_count == 2


@patch("src.coindcx_client._signed_post")
def test_create_order_never_retries_a_failed_call(mock_post):
    """A failed create_order whose response was lost but which actually
    succeeded server-side would place a SECOND order on retry — a real
    double-submission risk unlike a re-read. This must fail once, not
    retry, unlike every other coindcx_client function."""
    mock_post.side_effect = requests.ConnectionError("blip")

    try:
        create_order("BTCINR", "buy", 1.0)
        assert False, "expected ConnectionError"
    except requests.ConnectionError:
        pass

    assert mock_post.call_count == 1
