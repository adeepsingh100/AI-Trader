"""Fill simulation. Commission reuses execution/paper.py's exact fee
formula (imported directly — live's fee behavior is untouched). Everything
else (spread/slippage/partial fills/rejections/retry-via-resubmit) is new,
deterministic, and configurable. Never imports src.db.models or
src.coindcx_client — fully in-memory, enforced by
tests/test_backtest_execution_simulator.py patching requests.get/post to
raise if called from this module."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.execution.paper import fees as compute_fees
from src.backtest.order_manager import Order, OrderType
from src.config import (
    BACKTEST_MAX_FILL_PCT_OF_BAR_VOLUME,
    BACKTEST_MIN_NOTIONAL_INR,
    BACKTEST_SLIPPAGE_BPS,
    BACKTEST_SPREAD_BPS,
)


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    fill_price: float
    fees: float
    slippage_cost: float
    status: str  # "filled" | "partial" | "rejected"
    rejection_reason: str | None = None


def _slippage_and_spread_price(price: float, side: str) -> tuple[float, float]:
    """Directional slippage (against the trader, mirrors
    execution/paper.py's SLIPPAGE_BPS convention with an independently
    configurable value) + a synthetic half-spread cost — CoinDCX exposes
    no historical order-book snapshots, so real book-depth replay isn't
    possible; this is a documented approximation. Returns
    (fill_price, adverse_cost_per_unit)."""
    slip = price * (BACKTEST_SLIPPAGE_BPS / 10_000)
    spread = price * (BACKTEST_SPREAD_BPS / 10_000) / 2
    adverse = slip + spread
    fill_price = price + adverse if side == "buy" else price - adverse
    return fill_price, adverse


def _liquidity_cap(requested_qty: float, bar_volume: float) -> float:
    max_qty = bar_volume * (BACKTEST_MAX_FILL_PCT_OF_BAR_VOLUME / 100)
    return max(0.0, min(requested_qty, max_qty))


def execute_market_order(symbol: str, side: str, qty: float, reference_price: float, bar_volume: float) -> Fill:
    """Fills against reference_price — the caller's ticker-price proxy
    (the still-forming tick bar's open, mirroring live's get_ticker()
    last_price). Deterministic given the same inputs; a partial/rejected
    fill is a real, distinct outcome, never silently rounded up to full."""
    fill_price, slip_per_unit = _slippage_and_spread_price(reference_price, side)
    fillable_qty = _liquidity_cap(qty, bar_volume)
    if fillable_qty <= 0 or fill_price * fillable_qty < BACKTEST_MIN_NOTIONAL_INR:
        return Fill(symbol, side, 0.0, fill_price, 0.0, 0.0, "rejected", "below_min_notional_or_no_liquidity")
    fee_amount = compute_fees(fill_price * fillable_qty, side)
    slippage_cost = slip_per_unit * fillable_qty
    status = "filled" if fillable_qty >= qty else "partial"
    return Fill(symbol, side, fillable_qty, fill_price, fee_amount, slippage_cost, status)


def check_resting_order_fill(order: Order, bar: dict) -> Fill | None:
    """Standard OHLC-range-crossing fill logic for resting Limit/Stop/
    StopLimit/TrailingStop orders — a real, generic capability. Live never
    issues these (CoinDCX spot has no exchange-side resting order; SL/TP
    is polled market-order exits, handled separately by engine.py's
    risk-check cadence, not through this path) — this exists for testing a
    different, limit/stop-based strategy against the engine. None if the
    bar's range doesn't cross the order's trigger — no fill this bar."""
    high, low = float(bar["high"]), float(bar["low"])
    remaining = order.qty - order.filled_qty

    if order.order_type == OrderType.LIMIT:
        crossed = (order.side == "buy" and low <= order.limit_price) or (
            order.side == "sell" and high >= order.limit_price
        )
        if not crossed:
            return None
        trigger_price = order.limit_price

    elif order.order_type == OrderType.STOP:
        crossed = (order.side == "buy" and high >= order.stop_price) or (
            order.side == "sell" and low <= order.stop_price
        )
        if not crossed:
            return None
        trigger_price = order.stop_price

    elif order.order_type == OrderType.STOP_LIMIT:
        stop_crossed = (order.side == "buy" and high >= order.stop_price) or (
            order.side == "sell" and low <= order.stop_price
        )
        limit_crossed = (order.side == "buy" and low <= order.limit_price) or (
            order.side == "sell" and high >= order.limit_price
        )
        if not (stop_crossed and limit_crossed):
            return None
        trigger_price = order.limit_price

    elif order.order_type == OrderType.TRAILING_STOP:
        # Crossing is checked against the trail level carried in FROM
        # BEFORE this bar, then this bar's favorable extreme folds into
        # the trail for the NEXT bar — checking against a level this same
        # bar just extended would let one wide up-bar's own high trigger
        # an immediate stop-out via its own low, which isn't how a
        # trailing stop behaves in practice.
        trail_stop = None
        crossed = False
        if order.trail_extreme is not None:
            trail_stop = (
                order.trail_extreme * (1 - order.trail_pct)
                if order.side == "sell"
                else order.trail_extreme * (1 + order.trail_pct)
            )
            crossed = (order.side == "sell" and low <= trail_stop) or (order.side == "buy" and high >= trail_stop)

        favorable_extreme = high if order.side == "sell" else low
        if order.trail_extreme is None or (
            (order.side == "sell" and favorable_extreme > order.trail_extreme)
            or (order.side == "buy" and favorable_extreme < order.trail_extreme)
        ):
            order.trail_extreme = favorable_extreme

        if not crossed:
            return None
        trigger_price = trail_stop

    else:
        return None

    fill_price, slip_per_unit = _slippage_and_spread_price(trigger_price, order.side)
    fillable_qty = _liquidity_cap(remaining, float(bar["volume"]))
    if fillable_qty <= 0:
        return None
    fee_amount = compute_fees(fill_price * fillable_qty, order.side)
    slippage_cost = slip_per_unit * fillable_qty
    status = "filled" if fillable_qty >= remaining else "partial"
    return Fill(order.symbol, order.side, fillable_qty, fill_price, fee_amount, slippage_cost, status)
