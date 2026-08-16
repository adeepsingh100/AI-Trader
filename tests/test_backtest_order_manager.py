from datetime import datetime, timezone

import pytest

from src.backtest.order_manager import Order, OrderManager, OrderStatus, OrderType


def _order(order_type=OrderType.LIMIT, expiry_bars=5, submitted_bar_index=0):
    return Order(
        symbol="BTCINR",
        side="buy",
        order_type=order_type,
        qty=1.0,
        submitted_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        submitted_bar_index=submitted_bar_index,
        limit_price=100.0,
        expiry_bars=expiry_bars,
    )


def test_market_orders_are_rejected_from_order_manager():
    manager = OrderManager()
    with pytest.raises(ValueError):
        manager.submit(_order(order_type=OrderType.MARKET))


def test_submit_and_resting_orders_roundtrip():
    manager = OrderManager()
    order = _order()
    manager.submit(order)
    assert manager.resting_orders() == [order]
    assert manager.resting_orders(symbol="BTCINR") == [order]
    assert manager.resting_orders(symbol="ETHINR") == []


def test_cancel_removes_from_resting():
    manager = OrderManager()
    order = _order()
    manager.submit(order)
    manager.cancel(order)
    assert order.status == OrderStatus.CANCELLED
    assert manager.resting_orders() == []


def test_expire_stale_marks_orders_past_expiry_bars():
    manager = OrderManager()
    order = _order(expiry_bars=3, submitted_bar_index=0)
    manager.submit(order)

    assert manager.expire_stale(current_bar_index=2) == []  # not yet
    assert order.status == OrderStatus.SUBMITTED

    expired = manager.expire_stale(current_bar_index=3)
    assert expired == [order]
    assert order.status == OrderStatus.EXPIRED


def test_purge_settled_drops_non_resting_orders():
    manager = OrderManager()
    resting = _order()
    filled = _order()
    manager.submit(resting)
    manager.submit(filled)
    filled.status = OrderStatus.FILLED  # simulate the caller marking a fill externally

    manager.purge_settled()

    assert manager.resting_orders() == [resting]
