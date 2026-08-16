"""Order lifecycle: submit/track/expire/cancel. Market orders never enter
this manager — they resolve same-tick via
execution_simulator.execute_market_order. Limit/Stop/StopLimit/
TrailingStop orders rest here until a later bar's OHLC range crosses their
trigger, they expire (BACKTEST_ORDER_EXPIRY_BARS), or they're cancelled.
Live only ever issues market orders (CoinDCX spot has no exchange-side
resting order) — these richer types are a real, generic capability for
testing a different strategy against this engine, not something the
default parity backtest of the current live strategy exercises."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.config import BACKTEST_ORDER_EXPIRY_BARS


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    order_type: OrderType
    qty: float
    submitted_time: datetime
    submitted_bar_index: int
    limit_price: float | None = None
    stop_price: float | None = None
    trail_pct: float | None = None
    expiry_bars: int = BACKTEST_ORDER_EXPIRY_BARS
    status: OrderStatus = OrderStatus.SUBMITTED
    filled_qty: float = 0.0
    reason: str | None = None
    trail_extreme: float | None = field(default=None, repr=False)

    def is_resting(self) -> bool:
        return self.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL)


class OrderManager:
    def __init__(self):
        self._resting: list[Order] = []

    def submit(self, order: Order) -> None:
        if order.order_type == OrderType.MARKET:
            raise ValueError("market orders resolve same-tick, do not go through OrderManager")
        self._resting.append(order)

    def resting_orders(self, symbol: str | None = None) -> list[Order]:
        orders = [o for o in self._resting if o.is_resting()]
        return [o for o in orders if o.symbol == symbol] if symbol else orders

    def cancel(self, order: Order) -> None:
        order.status = OrderStatus.CANCELLED

    def expire_stale(self, current_bar_index: int) -> list[Order]:
        expired = []
        for o in self.resting_orders():
            if current_bar_index - o.submitted_bar_index >= o.expiry_bars:
                o.status = OrderStatus.EXPIRED
                expired.append(o)
        return expired

    def purge_settled(self) -> None:
        """Drops filled/rejected/expired/cancelled orders — keeps
        resting_orders() cheap over a long run."""
        self._resting = [o for o in self._resting if o.is_resting()]
