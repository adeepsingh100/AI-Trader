"""One full cycle: data -> signal -> risk -> execute -> log. Single
invocation, runs to completion, exits — see PROJECT_SPEC.md §5.

Spot-only: a "sell" signal closes an existing held position for that
symbol (there's no shorting on CoinDCX spot). A "buy" opens one if the
Risk Manager clears it. Circuit breaker is checked before every buy,
not just once at cycle start, since a loss realized earlier in the same
cycle must stop later buys in that same cycle."""

from __future__ import annotations

from src.agents.data_agent import get_market_snapshot
from src.agents.execution.paper import PaperExecutionAgent
from src.agents.execution.real import RealExecutionAgent
from src.agents.risk_manager import circuit_breaker_triggered, evaluate, target_hit, today_ist
from src.agents.signal_agent import get_signal
from src.db import models

MODE = "paper"


def _empty_daily_pnl() -> dict:
    return {"realized_pnl": 0, "trades_count": 0, "circuit_breaker_triggered": False}


def _record_close(mode: str, capital_config: dict, trade: dict, fill: dict, daily_pnl: dict):
    pnl = (fill["fill_price"] - trade["entry_price"]) * trade["qty"] - fill["fees"] - trade["fees"]
    models.close_trade(trade["id"], fill["fill_price"], pnl)

    updated = {
        "realized_pnl": daily_pnl["realized_pnl"] + pnl,
        "trades_count": daily_pnl["trades_count"] + 1,
        "circuit_breaker_triggered": daily_pnl["circuit_breaker_triggered"],
    }
    updated["circuit_breaker_triggered"] = circuit_breaker_triggered(updated, capital_config)
    models.upsert_daily_pnl(
        today_ist(),
        mode,
        realized_pnl=updated["realized_pnl"],
        trades_count=updated["trades_count"],
        target_hit=target_hit(updated, capital_config),
        circuit_breaker_triggered=updated["circuit_breaker_triggered"],
    )
    return pnl, updated


def run_cycle(mode: str = MODE, execution_agent=None, n_symbols: int = 10) -> dict:
    capital_config = models.get_capital_config(mode)

    if mode == "real":
        # Real mode is expected to sit unconfigured for a long time (no
        # capital_config, no promoted version) while paper trading earns
        # its way to promotion — both are "not ready yet", not errors,
        # so the cron job no-ops instead of failing every 10 minutes.
        if capital_config is None:
            return {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "no_capital_config"}

        if capital_config.get("paused"):
            return {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "paused"}

        # real only ever trades the version that's actually been promoted —
        # NOT just the newest strategy_versions row, since evolution keeps
        # minting new (unvetted) paper versions after a promotion too.
        version = models.get_latest_promoted_version()
        if version is None:
            models.log_agent_event(
                "orchestrator", "info", "real mode: no promoted strategy version yet, skipping"
            )
            return {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "no_promoted_version"}
    else:
        if capital_config is None:
            raise RuntimeError(f"no capital_config row for mode={mode!r} — insert one first")
        if capital_config.get("paused"):
            return {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "paused"}
        version = models.get_latest_version()
        if version is None:
            raise RuntimeError("no strategy_versions row — create one first")

    if execution_agent is None:
        execution_agent = PaperExecutionAgent() if mode == "paper" else RealExecutionAgent()

    daily_pnl = models.get_daily_pnl(today_ist(), mode) or _empty_daily_pnl()

    if circuit_breaker_triggered(daily_pnl, capital_config):
        execution_agent.flatten_all(mode)
        return {"opened": [], "closed": [], "circuit_breaker": True}

    open_trades = models.get_open_trades(mode)
    open_by_symbol = {t["symbol"]: t for t in open_trades}
    snapshot = get_market_snapshot(n_symbols)

    opened, closed = [], []
    for market in snapshot:
        if circuit_breaker_triggered(daily_pnl, capital_config):
            execution_agent.flatten_all(mode)
            break

        signal, usage_events = get_signal(market, version["prompt_text"])
        models.log_model_usage(usage_events)

        held = open_by_symbol.get(market["symbol"])

        if signal["direction"] == "sell" and held is not None:
            fill = execution_agent.place_order(
                market["symbol"], "sell", held["qty"], market["last_price"]
            )
            _, daily_pnl = _record_close(mode, capital_config, held, fill, daily_pnl)
            closed.append(held)
            open_trades.remove(held)
            del open_by_symbol[market["symbol"]]
            continue

        if signal["direction"] != "buy" or held is not None:
            continue  # flat, sell-with-nothing-held, or already holding (no pyramiding)

        decision = evaluate(capital_config, daily_pnl, open_trades, market["last_price"])
        if decision.action != "size":
            continue

        fill = execution_agent.place_order(
            market["symbol"], "buy", decision.qty, market["last_price"]
        )
        trade = models.open_trade(
            mode=mode,
            version_id=version["id"],
            symbol=market["symbol"],
            side="buy",
            qty=decision.qty,
            entry_price=fill["fill_price"],
            fees=fill["fees"],
            reasoning_text=signal["reasoning"],
        )
        models.log_agent_event("orchestrator", "info", f"opened buy {market['symbol']}")
        opened.append(trade)
        open_trades.append(trade)
        open_by_symbol[market["symbol"]] = trade

    return {
        "opened": opened,
        "closed": closed,
        "circuit_breaker": circuit_breaker_triggered(daily_pnl, capital_config),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=MODE, choices=["paper", "real"])
    args = parser.parse_args()
    print(run_cycle(mode=args.mode))
