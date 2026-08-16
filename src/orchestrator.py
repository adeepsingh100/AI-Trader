"""One full cycle: scan -> feature engine -> opportunity score -> top
candidates -> LLM validation -> risk -> execute -> log. Single invocation,
runs to completion, exits — see PROJECT_SPEC.md §5.

Quant-first: OpportunityScorer (src/features/) deterministically scores
every scanned symbol, zero LLM calls. Only the top-scoring not-held
candidates (TOP_N_CANDIDATES) and held positions whose score has fallen
below EXIT_SCORE_THRESHOLD go to the LLM for accept/reject validation —
the LLM never sees raw candles and never picks direction on its own.

Spot-only: an accepted exit closes an existing held position for that
symbol (there's no shorting on CoinDCX spot). An accepted entry opens one
if the Risk Manager clears it. Circuit breaker is checked before every
buy, not just once at cycle start, since a loss realized earlier in the
same cycle must stop later buys in that same cycle.

Stop-loss/take-profit (from the active version's params_json) is swept
every cycle too, and doesn't wait on the LLM to say "sell" — see
run_risk_check() below, which does only that sweep with no LLM calls,
meant for a tighter cron than the full LLM validation cycle."""

from __future__ import annotations

from src.agents.data_agent import get_market_snapshot
from src.agents.execution.paper import PaperExecutionAgent
from src.agents.execution.real import RealExecutionAgent
from src.agents.risk_manager import (
    circuit_breaker_triggered,
    evaluate,
    exit_reason,
    target_hit,
    today_ist,
)
from src.agents.signal_agent import validate_opportunity
from src.coindcx_client import get_ticker
from src.config import EXIT_SCORE_THRESHOLD, TIMEFRAME_WEIGHTS
from src.db import models
from src.features.feature_engine import compute_multi_timeframe_features
from src.features.opportunity_scorer import score_opportunity, select_top_candidates

MODE = "paper"

# The blend across timeframes weights every configured timeframe, but a
# few point-in-time context fields in the LLM summary (support/resistance,
# volatility label, ADX) can't be blended — they're read from whichever
# configured timeframe carries the most weight, since that's the one the
# scoring itself trusts most. Dynamic, not hardcoded to a specific string,
# so re-weighting TIMEFRAME_WEIGHTS in config also moves this.
_PRIMARY_TIMEFRAME = max(TIMEFRAME_WEIGHTS, key=TIMEFRAME_WEIGHTS.get)


def _empty_daily_pnl() -> dict:
    return {"realized_pnl": 0, "trades_count": 0, "circuit_breaker_triggered": False}


def _sweep_stop_loss_take_profit(
    mode: str,
    capital_config: dict,
    version: dict,
    open_trades: list[dict],
    open_by_symbol: dict,
    daily_pnl: dict,
    execution_agent,
) -> tuple[list[dict], dict]:
    if not open_trades:
        return [], daily_pnl

    prices = {t["market"]: float(t["last_price"]) for t in get_ticker()}
    params_json = version.get("params_json") or {}

    closed = []
    for trade in list(open_trades):
        price = prices.get(trade["symbol"])
        if price is None:
            continue
        reason = exit_reason(trade["entry_price"], price, params_json)
        if reason is None:
            continue

        fill = execution_agent.place_order(trade["symbol"], "sell", trade["qty"], price)
        _, daily_pnl = _record_close(mode, capital_config, trade, fill, daily_pnl)
        models.log_agent_event("orchestrator", "info", f"{reason} exit {trade['symbol']}")
        closed.append(trade)
        open_trades.remove(trade)
        del open_by_symbol[trade["symbol"]]

    return closed, daily_pnl


def _opportunity_summary(record: dict, held: dict | None = None) -> dict:
    """Curated digest for the LLM — never the raw multi-timeframe feature
    dump (160+ floats), which would strain the token budget for no gain
    over what a human/LLM actually needs to judge the call."""
    primary = record["features_by_tf"].get(_PRIMARY_TIMEFRAME) or {}
    summary = {
        "symbol": record["symbol"],
        "last_price": record["market"]["last_price"],
        "opportunity_score": record["opportunity_score"],
        "sub_scores": {
            "trend": record["trend_score"],
            "momentum": record["momentum_score"],
            "volume": record["volume_score"],
            "volatility": record["volatility_score"],
            "risk": record["risk_score"],
        },
        "volatility_label": primary.get("volatility_regime"),
        "support": primary.get("support"),
        "resistance": primary.get("resistance"),
        "distance_from_resistance_pct": primary.get("distance_from_resistance_pct"),
        "volume_spike": primary.get("volume_spike"),
        "adx": primary.get("adx"),
        "di_plus": primary.get("di_plus"),
        "di_minus": primary.get("di_minus"),
    }
    if held is not None:
        entry_price = held["entry_price"]
        last_price = record["market"]["last_price"]
        summary["held_position"] = {
            "entry_price": entry_price,
            "qty": held["qty"],
            "unrealized_pnl_pct": (last_price - entry_price) / entry_price * 100 if entry_price else None,
        }
    return summary


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

    opened, closed = [], []
    stopped, daily_pnl = _sweep_stop_loss_take_profit(
        mode, capital_config, version, open_trades, open_by_symbol, daily_pnl, execution_agent
    )
    closed.extend(stopped)

    if circuit_breaker_triggered(daily_pnl, capital_config):
        execution_agent.flatten_all(mode)
        return {"opened": opened, "closed": closed, "circuit_breaker": True}

    snapshot = get_market_snapshot(n_symbols)

    # Pass 1: pure, no LLM, no side effects — score every scanned symbol.
    scored = []
    for market in snapshot:
        features_by_tf = compute_multi_timeframe_features(market["candles_by_timeframe"])
        scores = score_opportunity(features_by_tf)
        scored.append(
            {"symbol": market["symbol"], "market": market, "features_by_tf": features_by_tf, **scores}
        )

    not_held = [r for r in scored if r["symbol"] not in open_by_symbol]
    candidate_symbols = {r["symbol"] for r in select_top_candidates(not_held)}

    # Pass 2: LLM validation only for entry candidates and held positions
    # whose score has deteriorated past the exit threshold — everyone else
    # is logged with no LLM call at all.
    for record in scored:
        if circuit_breaker_triggered(daily_pnl, capital_config):
            execution_agent.flatten_all(mode)
            break

        symbol = record["symbol"]
        market = record["market"]
        held = open_by_symbol.get(symbol)
        opportunity_score = record["opportunity_score"]

        llm_decision = llm_reasoning = llm_raw_response = risk_manager_result = None
        final_decision, reason = "hold", None

        if held is None and symbol in candidate_symbols:
            summary = _opportunity_summary(record)
            verdict, usage_events = validate_opportunity(summary, version["prompt_text"], context="entry")
            models.log_model_usage(usage_events)
            llm_decision = verdict.get("decision")
            llm_reasoning = verdict.get("reasoning")
            llm_raw_response = verdict

            if llm_decision == "accept":
                decision = evaluate(capital_config, daily_pnl, open_trades, market["last_price"])
                risk_manager_result = decision.action
                if decision.action == "size":
                    fill = execution_agent.place_order(symbol, "buy", decision.qty, market["last_price"])
                    trade = models.open_trade(
                        mode=mode,
                        version_id=version["id"],
                        symbol=symbol,
                        side="buy",
                        qty=decision.qty,
                        entry_price=fill["fill_price"],
                        fees=fill["fees"],
                        reasoning_text=llm_reasoning,
                    )
                    models.log_agent_event("orchestrator", "info", f"opened buy {symbol}")
                    opened.append(trade)
                    open_trades.append(trade)
                    open_by_symbol[symbol] = trade
                    final_decision, reason = "buy", llm_reasoning
                else:
                    reason = f"risk_manager blocked: {decision.action}"
            else:
                reason = llm_reasoning or "llm rejected entry"

        elif held is not None and opportunity_score is not None and opportunity_score < EXIT_SCORE_THRESHOLD:
            summary = _opportunity_summary(record, held=held)
            verdict, usage_events = validate_opportunity(summary, version["prompt_text"], context="exit")
            models.log_model_usage(usage_events)
            llm_decision = verdict.get("decision")
            llm_reasoning = verdict.get("reasoning")
            llm_raw_response = verdict

            if llm_decision == "accept":
                fill = execution_agent.place_order(symbol, "sell", held["qty"], market["last_price"])
                _, daily_pnl = _record_close(mode, capital_config, held, fill, daily_pnl)
                closed.append(held)
                open_trades.remove(held)
                del open_by_symbol[symbol]
                final_decision, reason = "sell", llm_reasoning
            else:
                reason = llm_reasoning or "llm rejected exit"
        else:
            reason = "not_a_candidate" if held is None else "score_above_exit_threshold_or_unavailable"

        models.log_opportunity_evaluation(
            mode=mode,
            symbol=symbol,
            version_id=version["id"],
            features=record["features_by_tf"],
            trend_score=record["trend_score"],
            momentum_score=record["momentum_score"],
            volume_score=record["volume_score"],
            volatility_score=record["volatility_score"],
            risk_score=record["risk_score"],
            opportunity_score=opportunity_score,
            llm_decision=llm_decision,
            llm_reasoning=llm_reasoning,
            llm_raw_response=llm_raw_response,
            risk_manager_result=risk_manager_result,
            final_decision=final_decision,
            reason=reason,
        )

    return {
        "opened": opened,
        "closed": closed,
        "circuit_breaker": circuit_breaker_triggered(daily_pnl, capital_config),
    }


def run_risk_check(mode: str = MODE, execution_agent=None) -> dict:
    """No LLM calls, no market snapshot — just the circuit breaker and
    the stop-loss/take-profit sweep. Meant to run on a tighter cron than
    run_cycle (which is throttled by LLM budget), so a bad move gets cut
    off sooner than a full 10-minute wait."""
    capital_config = models.get_capital_config(mode)
    if capital_config is None or capital_config.get("paused"):
        return {"closed": [], "circuit_breaker": False, "skipped": "not_configured_or_paused"}

    version = (
        models.get_latest_promoted_version() if mode == "real" else models.get_latest_version()
    )
    if version is None:
        return {"closed": [], "circuit_breaker": False, "skipped": "no_version"}

    if execution_agent is None:
        execution_agent = PaperExecutionAgent() if mode == "paper" else RealExecutionAgent()

    daily_pnl = models.get_daily_pnl(today_ist(), mode) or _empty_daily_pnl()

    if circuit_breaker_triggered(daily_pnl, capital_config):
        execution_agent.flatten_all(mode)
        return {"closed": [], "circuit_breaker": True}

    open_trades = models.get_open_trades(mode)
    open_by_symbol = {t["symbol"]: t for t in open_trades}
    closed, daily_pnl = _sweep_stop_loss_take_profit(
        mode, capital_config, version, open_trades, open_by_symbol, daily_pnl, execution_agent
    )

    return {"closed": closed, "circuit_breaker": circuit_breaker_triggered(daily_pnl, capital_config)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=MODE, choices=["paper", "real"])
    parser.add_argument(
        "--risk-only",
        action="store_true",
        help="stop-loss/take-profit + circuit-breaker sweep only, no LLM calls",
    )
    args = parser.parse_args()
    print(run_risk_check(mode=args.mode) if args.risk_only else run_cycle(mode=args.mode))
