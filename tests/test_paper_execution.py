from unittest.mock import patch

from src.agents.execution.paper import (
    GST_PCT_ON_FEE,
    SELL_TDS_PCT,
    SLIPPAGE_BPS,
    TRADING_FEE_PCT,
    PaperExecutionAgent,
)
from src.execution_optimizer.optimizer import OrderContext


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


# --- Execution Optimizer integration (PROJECT_SPEC.md §3d) — additive,
# optional order_context kwarg; every test above (no order_context passed)
# must stay byte-identical, which they already are since the default is
# None and EXECUTION_OPTIMIZER_ENABLED defaults to false.


def test_place_order_ignores_order_context_when_optimizer_disabled():
    agent = PaperExecutionAgent()
    ctx = OrderContext("BTCINR", "buy", order_size=1000, spread_bps=50, bar_volume=1_000_000, recent_fill_rate=1.0)
    with patch("src.agents.execution.paper.EXECUTION_OPTIMIZER_ENABLED", False):
        fill = agent.place_order("BTCINR", "buy", qty=0.01, price=1_000_000, order_context=ctx)
    assert fill["order_type"] == "market"
    assert fill["fill_price"] == 1_000_000 * (1 + SLIPPAGE_BPS / 10_000)


def test_place_order_uses_limit_fill_when_optimizer_enabled_and_recommends_limit():
    agent = PaperExecutionAgent()
    ctx = OrderContext("BTCINR", "buy", order_size=1000, spread_bps=50, bar_volume=1_000_000, recent_fill_rate=1.0)
    with patch("src.agents.execution.paper.EXECUTION_OPTIMIZER_ENABLED", True), patch(
        "src.agents.execution.paper.random.random", return_value=0.0
    ):
        fill = agent.place_order("BTCINR", "buy", qty=0.01, price=1_000_000, order_context=ctx)
    assert fill["order_type"] == "limit"
    half_spread = 1_000_000 * (50 / 2 / 10_000)
    assert fill["fill_price"] == 1_000_000 - half_spread  # buy limit improves on price, doesn't cross full spread


def test_place_order_falls_back_to_market_when_limit_attempt_misses():
    agent = PaperExecutionAgent()
    ctx = OrderContext("BTCINR", "buy", order_size=1000, spread_bps=50, bar_volume=1_000_000, recent_fill_rate=0.3)
    with patch("src.agents.execution.paper.EXECUTION_OPTIMIZER_ENABLED", True), patch(
        "src.agents.execution.paper.random.random", return_value=0.99  # misses the 0.3 fill-probability draw
    ):
        fill = agent.place_order("BTCINR", "buy", qty=0.01, price=1_000_000, order_context=ctx)
    assert fill["order_type"] == "market"


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
