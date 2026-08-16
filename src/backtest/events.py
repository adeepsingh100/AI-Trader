"""Event types the BacktestEngine reactor dispatches on, plus a thin
EventQueue. Frozen dataclasses — events are facts about what happened,
never mutated after creation. Chronological ordering comes from
SimulationClock driving the outer loop (one clock tick fully drains the
queue before advancing), not from a priority queue here — stdlib
collections.deque is enough."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union


@dataclass(frozen=True)
class MarketEvent:
    """A symbol's data became visible at this simulated time (candles for
    every configured timeframe that are fully closed as of `time`)."""

    time: datetime
    symbol: str
    features_by_tf: dict[str, dict]
    last_price: float


@dataclass(frozen=True)
class SignalEvent:
    """Quant (and, if enabled, LLM) validation produced a decision for a
    symbol — accept-entry, accept-exit, or hold."""

    time: datetime
    symbol: str
    decision: str  # "enter" | "exit" | "hold"
    opportunity_score: float | None
    confidence: float | None
    market_regime: str | None
    reasoning: str | None = None


@dataclass(frozen=True)
class OrderEvent:
    time: datetime
    symbol: str
    side: str  # "buy" | "sell"
    order_type: str  # "market" | "limit" | "stop" | "stop_limit" | "trailing_stop"
    qty: float
    limit_price: float | None = None
    stop_price: float | None = None
    trail_pct: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class FillEvent:
    time: datetime
    symbol: str
    side: str
    qty: float
    fill_price: float
    fees: float
    slippage_cost: float
    status: str = "filled"  # "filled" | "partial"


@dataclass(frozen=True)
class PositionEvent:
    time: datetime
    symbol: str
    action: str  # "opened" | "closed"
    qty: float
    entry_price: float
    exit_price: float | None = None
    pnl: float | None = None
    exit_reason: str | None = None


@dataclass(frozen=True)
class PortfolioEvent:
    time: datetime
    cash: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    exposure_pct: float
    open_positions_count: int


@dataclass(frozen=True)
class RiskEvent:
    time: datetime
    action: str  # RiskDecision.action values, plus "circuit_breaker"
    symbol: str | None = None


@dataclass(frozen=True)
class TimeEvent:
    """Fired once per SimulationClock tick, before any MarketEvents for
    that tick — lets engine.py hook per-tick bookkeeping (mark-to-market,
    decision/risk-check cadence checks) without special-casing the loop."""

    time: datetime


Event = Union[
    MarketEvent, SignalEvent, OrderEvent, FillEvent, PositionEvent, PortfolioEvent, RiskEvent, TimeEvent
]


@dataclass
class EventQueue:
    _items: deque[Any] = field(default_factory=deque)

    def put(self, event: Event) -> None:
        self._items.append(event)

    def get(self) -> Event | None:
        return self._items.popleft() if self._items else None

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)
