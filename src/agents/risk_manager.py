"""Safety-critical. Enforces, in order: circuit breaker -> capital limit
-> position sizing. Daily profit target is tracked, not enforced — it's
a soft goal (see PROJECT_SPEC.md §2), so it never blocks a trade.

Pure functions: all state (capital_config, daily_pnl, open_trades) is
passed in and nothing here touches the DB, so this is fully unit
testable without a live Supabase connection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.agents.execution.paper import fees
from src.config import (
    EXIT_PARAM_SWEEP_MAX_PCT,
    EXIT_PARAM_SWEEP_MIN_PCT,
    EXPECTANCY_SPREAD_BPS,
    RISK_PER_TRADE_PCT,
    SLIPPAGE_BPS,
    STOP_LOSS_ATR_MULTIPLIER,
    TAKE_PROFIT_ATR_MULTIPLIER,
)
from src.portfolio.capital_allocation import compute_dynamic_size
from src.portfolio.intelligence import evaluate_candidate_impact
from src.utils import clamp

TRADING_DAY_TZ = ZoneInfo("Asia/Kolkata")


def today_ist() -> date:
    return datetime.now(TRADING_DAY_TZ).date()


@dataclass
class RiskDecision:
    action: str  # "size" | "block_circuit_breaker" | "block_max_positions" | "block_capital_limit" | "block_concentration_limit"
    qty: float | None = None


def circuit_breaker_triggered(daily_pnl: dict | None, capital_config: dict) -> bool:
    if daily_pnl is None:
        return False
    if daily_pnl.get("circuit_breaker_triggered"):
        return True
    realized_loss = -daily_pnl["realized_pnl"]
    return realized_loss >= capital_config["max_daily_loss"]


def target_hit(daily_pnl: dict | None, capital_config: dict) -> bool:
    if daily_pnl is None:
        return False
    return daily_pnl["realized_pnl"] >= capital_config["daily_profit_target"]


def committed_capital(open_trades: list[dict]) -> float:
    return sum(t["qty"] * t["entry_price"] for t in open_trades)


def exit_reason(
    entry_price: float,
    last_price: float,
    params_json: dict,
    stored_stop_loss_price: float | None = None,
    stored_take_profit_price: float | None = None,
) -> str | None:
    """stop_loss_pct/take_profit_pct from the active strategy version's
    params_json, as a decimal fraction of entry price (0.02 = 2%). Checked
    independently of the LLM signal so a hit exits immediately, not only
    when the LLM happens to say "sell" for that symbol in a given cycle.

    When params_json has a leg configured, behavior is exactly as before:
    live-recomputed every sweep from the CURRENT version's params, so a
    params_json update retroactively protects already-open positions.
    `stored_stop_loss_price`/`stored_take_profit_price` (new, additive,
    optional — every existing call site omitting them is unaffected) are
    that specific trade's own frozen-at-entry price (resolve_exit_params'
    ATR fallback, stored on the trade row at open) — used ONLY when
    params_json has no value for that leg, so a trade opened without a
    configured stop still has one enforced, instead of exit_reason()
    silently skipping the check (the "no stop = unbounded risk" gap
    statistics.py's _assess_risk already flags as "too_aggressive")."""
    if entry_price <= 0 or last_price <= 0:
        return None
    change = (last_price - entry_price) / entry_price

    stop_loss_pct = params_json.get("stop_loss_pct")
    if stop_loss_pct and change <= -abs(stop_loss_pct):
        return "stop_loss"
    if not stop_loss_pct and stored_stop_loss_price and last_price <= stored_stop_loss_price:
        return "stop_loss"

    take_profit_pct = params_json.get("take_profit_pct")
    if take_profit_pct and change >= abs(take_profit_pct):
        return "take_profit"
    if not take_profit_pct and stored_take_profit_price and last_price >= stored_take_profit_price:
        return "take_profit"

    return None


def resolve_exit_params(params_json: dict, atr_pct: float | None) -> tuple[float | None, float | None]:
    """stop_loss_pct/take_profit_pct, preferring params_json's
    evidence-validated value (already cleared the walk-forward/bootstrap/
    fitness gate in src/learning/simulation.py — primary, unchanged) and
    falling back to an ATR-derived value ONLY for a leg params_json doesn't
    configure. None for a leg with neither a configured value nor an
    atr_pct to derive one from — same "None, not a fabricated number"
    convention as feature_engine/opportunity_scorer throughout."""
    stop_loss_pct, take_profit_pct = params_json.get("stop_loss_pct"), params_json.get("take_profit_pct")

    if not stop_loss_pct and atr_pct is not None:
        stop_loss_pct = clamp(
            atr_pct / 100 * STOP_LOSS_ATR_MULTIPLIER, EXIT_PARAM_SWEEP_MIN_PCT, EXIT_PARAM_SWEEP_MAX_PCT
        )
    if not take_profit_pct and atr_pct is not None:
        take_profit_pct = clamp(
            atr_pct / 100 * TAKE_PROFIT_ATR_MULTIPLIER, EXIT_PARAM_SWEEP_MIN_PCT, EXIT_PARAM_SWEEP_MAX_PCT
        )

    return stop_loss_pct, take_profit_pct


def compute_net_expectancy_pct(
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    win_probability: float | None,
    spread_bps: float = EXPECTANCY_SPREAD_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
) -> dict | None:
    """Net Expectancy Gate: is this candidate worth trading once real costs
    are netted out? Pure percentage-of-notional math — every cost
    (TRADING_FEE_PCT/GST_PCT_ON_FEE/SELL_TDS_PCT via
    src/agents/execution/paper.py::fees(), the exact function that computes
    the real simulated fee) scales linearly with notional, so this needs no
    qty/entry_price, just a unit (1.0) notional. win_probability is the
    system's own existing probability estimate (calibrated final_confidence
    / 100) — no new estimator invented.

    None (not a fabricated number) when any required input is missing —
    the caller treats that as "can't evaluate, don't trade" same as every
    other None-degrades-to-no-trade gate in this codebase."""
    if stop_loss_pct is None or take_profit_pct is None or win_probability is None:
        return None
    if stop_loss_pct <= 0 or take_profit_pct <= 0:
        return None

    win_probability = clamp(win_probability, 0.0, 1.0)
    # sell fee computed against entry (not exit) notional for both outcomes —
    # the actual exit notional differs by +/- stop/target pct, a
    # second-order correction (fee_pct * target_pct, a fraction of a
    # fraction) too small to matter at this gate's precision.
    cost_pct = fees(1.0, "buy") + fees(1.0, "sell") + (spread_bps / 10_000) + (slippage_bps / 10_000)

    gross_profit_pct = take_profit_pct
    gross_loss_pct = stop_loss_pct
    net_profit_if_win = gross_profit_pct - cost_pct
    net_loss_if_lose = -gross_loss_pct - cost_pct
    net_expectancy_pct = win_probability * net_profit_if_win + (1 - win_probability) * net_loss_if_lose

    return {
        "gross_profit_pct": gross_profit_pct,
        "gross_loss_pct": gross_loss_pct,
        "cost_pct": cost_pct,
        "net_expectancy_pct": net_expectancy_pct,
        "risk_reward": gross_profit_pct / gross_loss_pct if gross_loss_pct else None,
        "win_probability": win_probability,
    }


def evaluate(
    capital_config: dict,
    daily_pnl: dict | None,
    open_trades: list[dict],
    last_price: float,
    symbol: str | None = None,
    portfolio_positions: list | None = None,
    price_history: dict | None = None,
    sizing_context: dict | None = None,
    stop_loss_pct: float | None = None,
) -> RiskDecision:
    """symbol/portfolio_positions/price_history/sizing_context/
    stop_loss_pct are new, additive, optional kwargs (Portfolio
    Intelligence + Capital Allocation + risk-based sizing) — omitted
    (every existing call site, every existing test), this function is
    byte-identical to before. Supplied, three independent things can
    happen: capital_config['sizing_mode'] == 'dynamic' switches the sizing
    formula (still gated by the exact same committed_capital ceiling
    below, never bypassing it); symbol + portfolio_positions +
    price_history together additionally run a concentration cap check via
    Portfolio Intelligence; stop_loss_pct additionally caps qty so no
    single trade risks more than RISK_PER_TRADE_PCT of capital_to_use if
    its stop is hit — a strict ADDITIONAL cap on top of whatever the
    flat/dynamic formula already computed, so it can only shrink qty,
    never grow it."""
    if circuit_breaker_triggered(daily_pnl, capital_config):
        return RiskDecision(action="block_circuit_breaker")

    if len(open_trades) >= capital_config["max_concurrent_positions"]:
        return RiskDecision(action="block_max_positions")

    base_trade_capital = capital_config["capital_to_use"] * (
        capital_config["position_size_pct"] / 100
    )

    if capital_config.get("sizing_mode") == "dynamic":
        trade_capital = compute_dynamic_size(base_trade_capital, **(sizing_context or {}))["trade_capital"]
    else:
        trade_capital = base_trade_capital

    if last_price <= 0 or trade_capital <= 0:
        return RiskDecision(action="block_capital_limit")

    if committed_capital(open_trades) + trade_capital > capital_config["capital_to_use"]:
        return RiskDecision(action="block_capital_limit")

    qty = trade_capital / last_price

    if stop_loss_pct:
        max_risk_amount = capital_config["capital_to_use"] * (RISK_PER_TRADE_PCT / 100)
        max_qty_by_risk = max_risk_amount / (stop_loss_pct * last_price)
        qty = min(qty, max_qty_by_risk)

    if symbol is not None and portfolio_positions is not None and price_history is not None:
        # capital_to_use stands in for "equity" here — live trading has no
        # mark-to-market PortfolioManager (that's backtest-only), so this
        # is the best available proxy: the TOTAL capital pool this mode is
        # allowed to deploy. Not net of committed capital — converting
        # cash to a position doesn't change total equity, and
        # concentration/sector caps are meant to be measured against the
        # whole pool (the standard "no single holding over N% of NAV"
        # convention), not against whatever happens to be still
        # uncommitted at this point in the cycle.
        equity = capital_config["capital_to_use"]
        impact = evaluate_candidate_impact(
            symbol,
            qty,
            last_price,
            portfolio_positions,
            price_history,
            equity,
            max_concurrent_positions=capital_config["max_concurrent_positions"],
        )
        if impact["exceeds_position_concentration_cap"] or impact["exceeds_sector_concentration_cap"]:
            return RiskDecision(action="block_concentration_limit")

    return RiskDecision(action="size", qty=qty)
