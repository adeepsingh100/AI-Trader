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
