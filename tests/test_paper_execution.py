from src.agents.execution.paper import SLIPPAGE_BPS, TAKER_FEE_PCT, PaperExecutionAgent


def test_place_order_buy_applies_slippage_and_fee():
    agent = PaperExecutionAgent()
    fill = agent.place_order("BTCINR", "buy", qty=0.01, price=1_000_000)

    expected_fill_price = 1_000_000 * (1 + SLIPPAGE_BPS / 10_000)
    assert fill["fill_price"] == expected_fill_price
    assert fill["fees"] == expected_fill_price * 0.01 * (TAKER_FEE_PCT / 100)


def test_place_order_sell_applies_slippage_against_price():
    agent = PaperExecutionAgent()
    fill = agent.place_order("BTCINR", "sell", qty=0.01, price=1_000_000)

    expected_fill_price = 1_000_000 * (1 - SLIPPAGE_BPS / 10_000)
    assert fill["fill_price"] == expected_fill_price
