from __future__ import annotations

import random

from src.agents.execution.base import ExecutionAgent
from src.backtest.order_manager import OrderType
from src.config import (
    EXECUTION_OPTIMIZER_ENABLED,
    GST_PCT_ON_FEE,
    SELL_TDS_PCT,
    SLIPPAGE_BPS,
    TRADING_FEE_PCT,
)
from src.execution_optimizer.optimizer import OrderContext, recommend


def sell_tds(notional: float) -> float:
    """1% TDS (Income Tax Act s.194S) on a sell's trade value — public,
    reused as-is by src/agents/execution/real.py so both paths compute the
    exact same tax figure off the same config constant."""
    return notional * (SELL_TDS_PCT / 100)


def fees(notional: float, side: str) -> float:
    """Public — reused as-is by src/backtest/execution_simulator.py for
    commission calc, same "make it public to reuse" precedent as
    opportunity_scorer.weighted_average."""
    trading_fee = notional * (TRADING_FEE_PCT / 100)
    total = trading_fee + trading_fee * (GST_PCT_ON_FEE / 100)
    if side == "sell":
        total += sell_tds(notional)
    return total


class PaperExecutionAgent(ExecutionAgent):
    def place_order(
        self, symbol: str, side: str, qty: float, price: float, order_context: OrderContext | None = None
    ) -> dict:
        """order_context is new, additive, optional (Execution Optimizer,
        PROJECT_SPEC.md §3d) — omitted (orchestrator.py's current call
        sites; live has no real spread/order-book data to build one from,
        since that fetch was deliberately dropped as a dead API call
        earlier in this codebase's history), behavior is unchanged: a
        flat-slippage simulated market fill, same as always.

        Supplied AND EXECUTION_OPTIMIZER_ENABLED, this attempts the
        recommended order type same-cycle: paper trading has no
        cross-cycle resting-order infrastructure (that's
        src/backtest/order_manager.py's job, built for the backtest
        engine's multi-tick event loop, not this synchronous per-cycle
        call) — a recommended LIMIT is modeled as filling at the
        half-spread-improved price with probability
        rec.estimated_fill_probability, a real but explicitly single-shot
        simplification, not persistent order resting. Falls through to
        the normal market fill on a miss or when MARKET is recommended."""
        if EXECUTION_OPTIMIZER_ENABLED and order_context is not None:
            rec = recommend(order_context)
            if rec.order_type == OrderType.LIMIT and random.random() < rec.estimated_fill_probability:
                half_spread = price * (order_context.spread_bps / 2 / 10_000)
                limit_price = price - half_spread if side == "buy" else price + half_spread
                fee_amount = fees(limit_price * qty, side)
                return {"fill_price": limit_price, "fees": fee_amount, "order_type": "limit"}

        slip = price * (SLIPPAGE_BPS / 10_000)
        fill_price = price + slip if side == "buy" else price - slip
        fee_amount = fees(fill_price * qty, side)
        return {"fill_price": fill_price, "fees": fee_amount, "order_type": "market"}

    def flatten_all(self, mode: str, strategy_type: str | None = None) -> list[dict]:
        from src.coindcx_client import get_ticker
        from src.db import models

        last_price = {t["market"]: float(t["last_price"]) for t in get_ticker()}

        closed = []
        for trade in models.get_open_trades(mode, strategy_type):
            price = last_price.get(trade["symbol"])
            if price is None:
                continue
            fill = self.place_order(trade["symbol"], "sell", trade["qty"], price)
            pnl = (fill["fill_price"] - trade["entry_price"]) * trade["qty"] - fill[
                "fees"
            ] - trade["fees"]
            models.close_trade(
                trade["id"], fill["fill_price"], pnl, status="flattened", exit_reason="circuit_breaker"
            )
            closed.append(trade)
        return closed
