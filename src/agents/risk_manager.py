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

TRADING_DAY_TZ = ZoneInfo("Asia/Kolkata")


def today_ist() -> date:
    return datetime.now(TRADING_DAY_TZ).date()


@dataclass
class RiskDecision:
    action: str  # "size" | "block_circuit_breaker" | "block_max_positions" | "block_capital_limit"
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
) -> RiskDecision:
    if circuit_breaker_triggered(daily_pnl, capital_config):
        return RiskDecision(action="block_circuit_breaker")

    if len(open_trades) >= capital_config["max_concurrent_positions"]:
        return RiskDecision(action="block_max_positions")

    trade_capital = capital_config["capital_to_use"] * (
        capital_config["position_size_pct"] / 100
    )
    if last_price <= 0 or trade_capital <= 0:
        return RiskDecision(action="block_capital_limit")

    if committed_capital(open_trades) + trade_capital > capital_config["capital_to_use"]:
        return RiskDecision(action="block_capital_limit")

    return RiskDecision(action="size", qty=trade_capital / last_price)
