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

from src.portfolio.capital_allocation import compute_dynamic_size
from src.portfolio.intelligence import evaluate_candidate_impact

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


def exit_reason(entry_price: float, last_price: float, params_json: dict) -> str | None:
    """stop_loss_pct/take_profit_pct from the active strategy version's
    params_json, as a decimal fraction of entry price (0.02 = 2%).
    Absent or falsy = that leg isn't enforced. Checked independently of
    the LLM signal so a hit exits immediately, not only when the LLM
    happens to say "sell" for that symbol in a given cycle."""
    if entry_price <= 0 or last_price <= 0:
        return None
    change = (last_price - entry_price) / entry_price

    stop_loss_pct = params_json.get("stop_loss_pct")
    if stop_loss_pct and change <= -abs(stop_loss_pct):
        return "stop_loss"

    take_profit_pct = params_json.get("take_profit_pct")
    if take_profit_pct and change >= abs(take_profit_pct):
        return "take_profit"

    return None


def evaluate(
    capital_config: dict,
    daily_pnl: dict | None,
    open_trades: list[dict],
    last_price: float,
    symbol: str | None = None,
    portfolio_positions: list | None = None,
    price_history: dict | None = None,
    sizing_context: dict | None = None,
) -> RiskDecision:
    """symbol/portfolio_positions/price_history/sizing_context are new,
    additive, optional kwargs (Portfolio Intelligence + Capital Allocation,
    PROJECT_SPEC.md §3d) — omitted (every existing call site, every
    existing test), this function is byte-identical to before. Supplied,
    two independent things can happen: capital_config['sizing_mode'] ==
    'dynamic' switches the sizing formula (still gated by the exact same
    committed_capital ceiling below, never bypassing it); symbol +
    portfolio_positions + price_history together additionally run a
    concentration cap check via Portfolio Intelligence."""
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
