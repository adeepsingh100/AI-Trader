from datetime import datetime, timezone
from unittest.mock import patch

from src.agents.execution.paper import fees as live_fees
from src.backtest.execution_simulator import execute_market_order, check_resting_order_fill
from src.backtest.order_manager import Order, OrderType


def _order(order_type, side="buy", limit_price=None, stop_price=None, trail_pct=None, qty=1.0):
    return Order(
        symbol="BTCINR",
        side=side,
        order_type=order_type,
        qty=qty,
        submitted_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        submitted_bar_index=0,
        limit_price=limit_price,
        stop_price=stop_price,
        trail_pct=trail_pct,
    )


# --- market orders ---


def test_execute_market_order_buy_fills_with_adverse_slippage_and_spread():
    fill = execute_market_order("BTCINR", "buy", qty=1.0, reference_price=1000.0, bar_volume=1000.0)
    assert fill.status == "filled"
    assert fill.fill_price > 1000.0  # buys pay up


def test_execute_market_order_sell_fills_below_reference():
    fill = execute_market_order("BTCINR", "sell", qty=1.0, reference_price=1000.0, bar_volume=1000.0)
    assert fill.status == "filled"
    assert fill.fill_price < 1000.0


def test_execute_market_order_commission_matches_live_fee_formula():
    fill = execute_market_order("BTCINR", "buy", qty=1.0, reference_price=1000.0, bar_volume=1000.0)
    expected_fees = live_fees(fill.fill_price * fill.qty, "buy")
    assert fill.fees == expected_fees


def test_execute_market_order_rejected_below_min_notional():
    fill = execute_market_order("BTCINR", "buy", qty=0.00001, reference_price=1000.0, bar_volume=1000.0)
    assert fill.status == "rejected"
    assert fill.qty == 0.0
    assert fill.rejection_reason == "below_min_notional_or_no_liquidity"


@patch("src.backtest.execution_simulator.BACKTEST_MAX_FILL_PCT_OF_BAR_VOLUME", 10)
@patch("src.backtest.execution_simulator.BACKTEST_MIN_NOTIONAL_INR", 0)
def test_execute_market_order_partial_fill_capped_by_bar_volume():
    fill = execute_market_order("BTCINR", "buy", qty=100.0, reference_price=1000.0, bar_volume=50.0)
    assert fill.status == "partial"
    assert fill.qty == 5.0  # 10% of 50


# --- resting orders: limit ---


def test_check_resting_order_fill_limit_buy_fills_when_low_crosses():
    order = _order(OrderType.LIMIT, side="buy", limit_price=100.0)
    bar = {"open": 105, "high": 106, "low": 99, "close": 101, "volume": 1000}
    fill = check_resting_order_fill(order, bar)
    assert fill is not None
    assert fill.status == "filled"


def test_check_resting_order_fill_limit_buy_no_fill_when_low_above_limit():
    order = _order(OrderType.LIMIT, side="buy", limit_price=90.0)
    bar = {"open": 105, "high": 106, "low": 99, "close": 101, "volume": 1000}
    assert check_resting_order_fill(order, bar) is None


def test_check_resting_order_fill_limit_sell_fills_when_high_crosses():
    order = _order(OrderType.LIMIT, side="sell", limit_price=110.0)
    bar = {"open": 105, "high": 111, "low": 104, "close": 106, "volume": 1000}
    fill = check_resting_order_fill(order, bar)
    assert fill is not None


# --- resting orders: stop ---


def test_check_resting_order_fill_stop_buy_fills_when_high_crosses():
    order = _order(OrderType.STOP, side="buy", stop_price=110.0)
    bar = {"open": 105, "high": 111, "low": 104, "close": 106, "volume": 1000}
    assert check_resting_order_fill(order, bar) is not None


def test_check_resting_order_fill_stop_sell_fills_when_low_crosses():
    order = _order(OrderType.STOP, side="sell", stop_price=90.0)
    bar = {"open": 105, "high": 106, "low": 89, "close": 100, "volume": 1000}
    assert check_resting_order_fill(order, bar) is not None


# --- resting orders: stop_limit ---


def test_check_resting_order_fill_stop_limit_needs_both_crossed():
    order = _order(OrderType.STOP_LIMIT, side="buy", stop_price=105.0, limit_price=108.0)
    # high crosses stop (105) but not the limit ceiling check (low<=108 true actually)
    bar = {"open": 100, "high": 106, "low": 99, "close": 101, "volume": 1000}
    fill = check_resting_order_fill(order, bar)
    assert fill is not None  # both conditions satisfied here


def test_check_resting_order_fill_stop_limit_stop_not_crossed_no_fill():
    order = _order(OrderType.STOP_LIMIT, side="buy", stop_price=200.0, limit_price=108.0)
    bar = {"open": 100, "high": 106, "low": 99, "close": 101, "volume": 1000}
    assert check_resting_order_fill(order, bar) is None


# --- resting orders: trailing stop ---


def test_check_resting_order_fill_trailing_stop_sell_tracks_favorable_high():
    order = _order(OrderType.TRAILING_STOP, side="sell", trail_pct=0.05)

    # First bar only seeds trail_extreme — nothing to trail from yet, so
    # no crossing check happens even if this bar's own range would
    # otherwise cross a trail computed from itself.
    bar1 = {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}
    assert check_resting_order_fill(order, bar1) is None
    assert order.trail_extreme == 101

    # Price rises to a new high (110) — trail_stop for THIS bar is still
    # based on the prior extreme (101 * 0.95 = 95.95), which this bar's
    # low (100) does not cross, so no same-bar false trigger from the new
    # high. trail_extreme updates to 110 for the next bar.
    bar2 = {"open": 101, "high": 110, "low": 100, "close": 109, "volume": 1000}
    assert check_resting_order_fill(order, bar2) is None
    assert order.trail_extreme == 110

    # trail_stop is now 110 * 0.95 = 104.5 (carried in from bar2's high) —
    # this bar's low (100) crosses it.
    bar3 = {"open": 109, "high": 111, "low": 100, "close": 103, "volume": 1000}
    fill = check_resting_order_fill(order, bar3)
    assert fill is not None
    assert fill.fill_price < 104.5  # sell fills at trail_stop minus adverse slippage/spread


def test_check_resting_order_fill_trailing_stop_first_bar_only_seeds_no_fill():
    """Nothing to trail from yet on the very first bar — even a bar whose
    own low would cross a trail computed from its own high must not fill,
    since that would mean one wide bar's own high can trigger its own
    stop-out (a same-bar look-ahead-shaped bug, not a real trailing stop)."""
    order = _order(OrderType.TRAILING_STOP, side="sell", trail_pct=0.10)
    bar = {"open": 100, "high": 100, "low": 85, "close": 90, "volume": 1000}
    assert check_resting_order_fill(order, bar) is None
    assert order.trail_extreme == 100


# --- network isolation: the module must never touch live network/DB ---


def test_execution_simulator_never_imports_network_or_db_modules():
    import src.backtest.execution_simulator as mod

    import_lines = [
        line.strip()
        for line in open(mod.__file__)
        if line.strip().startswith(("import ", "from ")) and "docstring" not in line
    ]
    assert not any("requests" in line for line in import_lines)
    assert not any("src.db" in line for line in import_lines)
    assert not any("coindcx_client" in line for line in import_lines)


@patch("requests.get")
@patch("requests.post")
def test_execute_market_order_makes_no_network_calls(mock_post, mock_get):
    execute_market_order("BTCINR", "buy", qty=1.0, reference_price=1000.0, bar_volume=1000.0)
    mock_get.assert_not_called()
    mock_post.assert_not_called()
