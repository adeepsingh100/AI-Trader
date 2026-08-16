"""Execution Optimizer — recommends MARKET vs. LIMIT (STOP/STOP_LIMIT for
callers that already know they want a stop) given spread/liquidity/
volatility/order-size/recent-fill-rate, and estimates fill probability/
cost/delay/slippage for that choice. See PROJECT_SPEC.md §3d.

Pure function, no DB/network access, matching execution_simulator.py's
isolation. Reuses src.backtest.order_manager.OrderType — no duplicate enum.

Scope: RealExecutionAgent stays fully untouched (market-only, per its own
documented unverified/inert status — see PROJECT_SPEC.md §2); a real
trade's recommendation is computed for the audit trail only, never acted
on. PaperExecutionAgent may act on a recommendation when
EXECUTION_OPTIMIZER_ENABLED=true (paper is simulated — safe to exercise a
LIMIT order via the existing backtest execution_simulator's fill logic)."""

from __future__ import annotations

from dataclasses import dataclass

from src.backtest.order_manager import OrderType
from src.config import (
    EXECUTION_OPTIMIZER_MIN_FILL_PROBABILITY,
    EXECUTION_OPTIMIZER_SPREAD_BPS_LIMIT_THRESHOLD,
)

REAL_EXECUTION_NOTE = "real execution is market-only pending promotion bar (see PROJECT_SPEC.md §2)"


@dataclass
class OrderContext:
    symbol: str
    side: str  # "buy" | "sell"
    order_size: float  # notional value of the intended order
    spread_bps: float
    bar_volume: float  # traded volume in the most recent bar, notional terms
    volatility_pct: float | None = None  # e.g. atr_pct from the Feature Engine
    recent_fill_rate: float | None = None  # 0-1, this symbol's recent limit-fill history


@dataclass
class ExecutionRecommendation:
    order_type: OrderType
    estimated_fill_probability: float
    estimated_cost_bps: float
    estimated_delay_bars: int
    estimated_slippage_bps: float
    reason: str


def _liquidity_slippage_bps(order_size: float, bar_volume: float) -> float:
    """Proportional to how much of the bar's volume the order would
    consume — same shape as execution_simulator.py's liquidity-cap logic,
    an independent estimate (this module never imports the simulator)."""
    if bar_volume <= 0:
        return 100.0  # no liquidity data — assume the worst, don't guess zero
    fraction = min(1.0, order_size / bar_volume)
    return fraction * 100  # 100% of bar volume consumed ~= 100bps of extra slippage, a simple linear proxy


def _limit_fill_probability(context: OrderContext) -> float:
    if context.recent_fill_rate is not None:
        return context.recent_fill_rate
    # No fill-rate history yet — fall back to a volatility-based heuristic:
    # a limit order is more likely to get crossed in a more volatile
    # market (price travels further per bar), capped well short of 1.0
    # since this is a heuristic, not a measured rate.
    volatility_pct = context.volatility_pct or 0.0
    return min(0.9, 0.4 + volatility_pct / 20)


def recommend(context: OrderContext) -> ExecutionRecommendation:
    fill_probability = _limit_fill_probability(context)
    slippage_bps = _liquidity_slippage_bps(context.order_size, context.bar_volume)

    prefer_limit = (
        context.spread_bps >= EXECUTION_OPTIMIZER_SPREAD_BPS_LIMIT_THRESHOLD
        and fill_probability >= EXECUTION_OPTIMIZER_MIN_FILL_PROBABILITY
    )

    if prefer_limit:
        return ExecutionRecommendation(
            order_type=OrderType.LIMIT,
            estimated_fill_probability=fill_probability,
            estimated_cost_bps=0.0,  # resting inside the spread, no cross cost
            estimated_delay_bars=max(1, int(5 * (1 - fill_probability))),
            estimated_slippage_bps=0.0,  # fixed limit price, no slippage if filled
            reason=(
                f"spread {context.spread_bps:.1f}bps >= threshold "
                f"{EXECUTION_OPTIMIZER_SPREAD_BPS_LIMIT_THRESHOLD:.1f}bps, "
                f"estimated fill probability {fill_probability:.0%} clears the floor"
            ),
        )

    return ExecutionRecommendation(
        order_type=OrderType.MARKET,
        estimated_fill_probability=1.0,
        estimated_cost_bps=context.spread_bps,
        estimated_delay_bars=0,
        estimated_slippage_bps=slippage_bps,
        reason=(
            f"spread {context.spread_bps:.1f}bps below threshold, or estimated limit fill "
            f"probability {fill_probability:.0%} too low — market fill preferred for certainty"
        ),
    )
