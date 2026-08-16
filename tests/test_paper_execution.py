from unittest.mock import patch

from src.agents.execution.paper import (
    GST_PCT_ON_FEE,
    SELL_TDS_PCT,
    SLIPPAGE_BPS,
    TRADING_FEE_PCT,
    PaperExecutionAgent,
)


def test_place_order_buy_applies_slippage_and_fee():
    agent = PaperExecutionAgent()
    fill = agent.place_order("BTCINR", "buy", qty=0.01, price=1_000_000)

    expected_fill_price = 1_000_000 * (1 + SLIPPAGE_BPS / 10_000)
    assert fill["fill_price"] == expected_fill_price

    notional = expected_fill_price * 0.01
    trading_fee = notional * (TRADING_FEE_PCT / 100)
    expected_fees = trading_fee + trading_fee * (GST_PCT_ON_FEE / 100)  # fee + 18% GST, no TDS
    assert fill["fees"] == expected_fees


def test_place_order_sell_applies_slippage_against_price():
    agent = PaperExecutionAgent()
    fill = agent.place_order("BTCINR", "sell", qty=0.01, price=1_000_000)

    expected_fill_price = 1_000_000 * (1 - SLIPPAGE_BPS / 10_000)
    assert fill["fill_price"] == expected_fill_price


def test_place_order_sell_fee_includes_tds_on_top_of_fee_and_gst():
    agent = PaperExecutionAgent()
    fill = agent.place_order("BTCINR", "sell", qty=0.01, price=1_000_000)

    notional = fill["fill_price"] * 0.01
    trading_fee = notional * (TRADING_FEE_PCT / 100)
    fee_plus_gst = trading_fee + trading_fee * (GST_PCT_ON_FEE / 100)
    expected_fees = fee_plus_gst + notional * (SELL_TDS_PCT / 100)
    assert fill["fees"] == expected_fees


def test_sell_fees_exceed_buy_fees_at_same_notional_due_to_tds():
    agent = PaperExecutionAgent()
    buy = agent.place_order("BTCINR", "buy", qty=0.01, price=1_000_000)
    sell = agent.place_order("BTCINR", "sell", qty=0.01, price=1_000_000)
    assert sell["fees"] > buy["fees"]


@patch("src.db.models")
@patch("src.coindcx_client.get_ticker")
def test_flatten_all_closes_every_open_trade_with_circuit_breaker_reason(mock_ticker, mock_models):
    held = {"id": 7, "symbol": "ETHINR", "qty": 0.5, "entry_price": 200_000, "fees": 1.0}
    mock_ticker.return_value = [{"market": "ETHINR", "last_price": 199_000}]
    mock_models.get_open_trades.return_value = [held]

    agent = PaperExecutionAgent()
    closed = agent.flatten_all("paper")

    mock_models.close_trade.assert_called_once()
    assert mock_models.close_trade.call_args.kwargs["exit_reason"] == "circuit_breaker"
    assert mock_models.close_trade.call_args.kwargs["status"] == "flattened"
    assert closed == [held]
