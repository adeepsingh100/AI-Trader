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

from datetime import datetime, timedelta, timezone

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
from src.config import (
    BUCKET_MODIFIER_CAP,
    BUCKET_MODIFIER_SENSITIVITY,
    EXIT_SCORE_THRESHOLD,
    LEARNING_CATCHUP_LOOKBACK_HOURS,
    MIN_FINAL_CONFIDENCE,
    RECENT_PERFORMANCE_LOOKBACK_TRADES,
    RECENT_STREAK_LOSS_MODIFIER_CAP,
    RECENT_STREAK_WIN_MODIFIER_CAP,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
)
from src.db import models
from src.features.feature_engine import compute_multi_timeframe_features
from src.features.opportunity_scorer import (
    PRIMARY_TIMEFRAME,
    score_opportunity,
    select_top_candidates,
)
from src.learning.confidence_calibration import calibrate_confidence
from src.learning.statistics import process_closed_trades, streaks
from src.learning.trade_memory import find_similar_trades

MODE = "paper"


def _empty_daily_pnl() -> dict:
    return {"realized_pnl": 0, "trades_count": 0, "circuit_breaker_triggered": False}


def _update_excursion(open_trades: list[dict], prices: dict) -> None:
    """Running max-favorable/max-adverse-excursion, updated with whatever
    ticker prices this sweep already fetched (zero new API calls). Writes
    only when a max actually moved. Runs on both cadences this function
    is called from (run_cycle 10 min, run_risk_check 5 min)."""
    for trade in open_trades:
        price = prices.get(trade["symbol"])
        entry = trade.get("entry_price")
        if price is None or not entry:
            continue
        favorable_pct = max(0.0, (price - entry) / entry * 100)
        adverse_pct = max(0.0, (entry - price) / entry * 100)
        prior_mfe, prior_mae = trade.get("mfe_pct") or 0, trade.get("mae_pct") or 0
        new_mfe, new_mae = max(prior_mfe, favorable_pct), max(prior_mae, adverse_pct)
        if new_mfe != prior_mfe or new_mae != prior_mae:
            models.update_trade_excursion(trade["id"], new_mfe, new_mae)
            trade["mfe_pct"], trade["mae_pct"] = new_mfe, new_mae


def _bucket_modifier(bucket_stats_by_value: dict, value: str | None, overall_win_rate: float | None) -> float | None:
    """Adaptive confidence chain (Step 7): how much better/worse this
    regime's or symbol's win rate is than the overall baseline, scaled
    and capped. None (no contribution) unless the bucket has enough
    samples to trust — never nudges confidence from a handful of trades."""
    if value is None or overall_win_rate is None:
        return None
    row = bucket_stats_by_value.get(value)
    if row is None:
        return None
    win_rate = row.get("win_rate")
    trades_count = row.get("trades_count") or 0
    if win_rate is None or trades_count < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return None
    raw = (win_rate - overall_win_rate) * BUCKET_MODIFIER_SENSITIVITY
    return max(-BUCKET_MODIFIER_CAP, min(BUCKET_MODIFIER_CAP, raw))


def _recent_performance_modifier(mode: str) -> float | None:
    """Adaptive confidence chain (Step 7): current win/loss streak over
    the last RECENT_PERFORMANCE_LOOKBACK_TRADES closed trades, scaled to
    the streak length and capped — deliberately asymmetric (a losing
    streak can suppress confidence more than a winning streak inflates
    it). Computed once per cycle, not per candidate."""
    since = datetime.now(timezone.utc) - timedelta(hours=LEARNING_CATCHUP_LOOKBACK_HOURS)
    recent = sorted(
        (t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None),
        key=lambda t: t["closed_at"],
    )[-RECENT_PERFORMANCE_LOOKBACK_TRADES:]
    if not recent:
        return None

    streak = streaks(recent)
    length_fraction = streak["current_streak_length"] / RECENT_PERFORMANCE_LOOKBACK_TRADES
    if streak["current_streak_type"] == "loss":
        return -min(RECENT_STREAK_LOSS_MODIFIER_CAP, length_fraction * RECENT_STREAK_LOSS_MODIFIER_CAP)
    if streak["current_streak_type"] == "win":
        return min(RECENT_STREAK_WIN_MODIFIER_CAP, length_fraction * RECENT_STREAK_WIN_MODIFIER_CAP)
    return None


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
    _update_excursion(open_trades, prices)
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
        _, daily_pnl = _record_close(mode, capital_config, trade, fill, daily_pnl, exit_reason_value=reason)
        models.log_agent_event("orchestrator", "info", f"{reason} exit {trade['symbol']}")
        closed.append(trade)
        open_trades.remove(trade)
        del open_by_symbol[trade["symbol"]]

    return closed, daily_pnl


def _opportunity_summary(record: dict, held: dict | None = None, historical_context: dict | None = None) -> dict:
    """Curated digest for the LLM — never the raw multi-timeframe feature
    dump (160+ floats), which would strain the token budget for no gain
    over what a human/LLM actually needs to judge the call."""
    primary = record["features_by_tf"].get(PRIMARY_TIMEFRAME) or {}
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
    if historical_context is not None:
        summary["historical_context"] = {
            "similar_trades_count": historical_context["count"],
            "win_rate": historical_context["win_rate"],
            "avg_profit_pct": historical_context["avg_profit_pct"],
            "avg_loss_pct": historical_context["avg_loss_pct"],
            "avg_holding_time_seconds": historical_context["avg_holding_time_seconds"],
        }
    return summary


def _record_close(
    mode: str, capital_config: dict, trade: dict, fill: dict, daily_pnl: dict, exit_reason_value: str | None = None
):
    pnl = (fill["fill_price"] - trade["entry_price"]) * trade["qty"] - fill["fees"] - trade["fees"]
    models.close_trade(trade["id"], fill["fill_price"], pnl, exit_reason=exit_reason_value)

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

    # Adaptive confidence chain (Step 7) inputs — fetched once per cycle,
    # not once per candidate (Step 13: cached/batched, not redundant
    # per-candidate queries). overall_win_rate uses the active version's
    # own learning_statistics bucket (already computed, already cached)
    # as the baseline regime/symbol win rates are compared against,
    # rather than a fresh full-trades scan every cycle.
    regime_stats = {r["dimension_value"]: r for r in models.get_learning_statistics(mode, dimension_type="market_regime")}
    symbol_stats = {r["dimension_value"]: r for r in models.get_learning_statistics(mode, dimension_type="symbol")}
    version_stats = {r["dimension_value"]: r for r in models.get_learning_statistics(mode, dimension_type="strategy_version")}
    overall_row = version_stats.get(str(version["id"]))
    overall_win_rate = overall_row.get("win_rate") if overall_row else None
    recent_performance_modifier = _recent_performance_modifier(mode)

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
        trade_id = None
        calibration_to_log = None

        if held is None and symbol in candidate_symbols:
            similar = find_similar_trades(record, market_regime=record["market_regime"], mode=mode)
            summary = _opportunity_summary(record, historical_context=similar)
            verdict, usage_events = validate_opportunity(summary, version["prompt_text"], context="entry")
            models.log_model_usage(usage_events)
            llm_decision = verdict.get("decision")
            llm_reasoning = verdict.get("reasoning")
            llm_raw_response = verdict

            historical_confidence = similar["win_rate"] * 100 if similar["win_rate"] is not None else None
            regime_modifier = _bucket_modifier(regime_stats, record["market_regime"], overall_win_rate)
            symbol_modifier = _bucket_modifier(symbol_stats, symbol, overall_win_rate)
            calibrated = calibrate_confidence(
                verdict.get("confidence"),
                historical_confidence,
                similar["count"],
                regime_modifier=regime_modifier,
                symbol_modifier=symbol_modifier,
                recent_performance_modifier=recent_performance_modifier,
            )
            calibration_to_log = {
                "ai_confidence": verdict.get("confidence"),
                "historical_confidence": historical_confidence,
                "ai_weight": calibrated["ai_weight_used"],
                "historical_weight": calibrated["historical_weight_used"],
                "final_confidence": calibrated["final_confidence"],
                "similar_trades_count": similar["count"],
                "regime_modifier": calibrated["regime_modifier"],
                "symbol_modifier": calibrated["symbol_modifier"],
                "recent_performance_modifier": calibrated["recent_performance_modifier"],
            }
            confidence_cleared = (
                calibrated["final_confidence"] is None or calibrated["final_confidence"] >= MIN_FINAL_CONFIDENCE
            )

            if llm_decision == "accept" and confidence_cleared:
                decision = evaluate(capital_config, daily_pnl, open_trades, market["last_price"])
                risk_manager_result = decision.action
                if decision.action == "size":
                    fill = execution_agent.place_order(symbol, "buy", decision.qty, market["last_price"])
                    params_json = version.get("params_json") or {}
                    stop_loss_pct = params_json.get("stop_loss_pct")
                    take_profit_pct = params_json.get("take_profit_pct")
                    entry_price = fill["fill_price"]
                    last_price = market["last_price"]
                    trade = models.open_trade(
                        mode=mode,
                        version_id=version["id"],
                        symbol=symbol,
                        side="buy",
                        qty=decision.qty,
                        entry_price=entry_price,
                        fees=fill["fees"],
                        reasoning_text=llm_reasoning,
                        stop_loss_price=entry_price * (1 - stop_loss_pct) if stop_loss_pct else None,
                        take_profit_price=entry_price * (1 + take_profit_pct) if take_profit_pct else None,
                        entry_slippage_pct=(entry_price - last_price) / last_price * 100 if last_price else None,
                        market_regime=record["market_regime"],
                    )
                    models.log_agent_event("orchestrator", "info", f"opened buy {symbol}")
                    opened.append(trade)
                    open_trades.append(trade)
                    open_by_symbol[symbol] = trade
                    final_decision, reason, trade_id = "buy", llm_reasoning, trade["id"]
                else:
                    reason = f"risk_manager blocked: {decision.action}"
            elif llm_decision == "accept":
                reason = f"confidence gated: {calibrated['final_confidence']:.1f} < {MIN_FINAL_CONFIDENCE}"
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
                _, daily_pnl = _record_close(mode, capital_config, held, fill, daily_pnl, exit_reason_value="ai_exit")
                closed.append(held)
                open_trades.remove(held)
                del open_by_symbol[symbol]
                final_decision, reason, trade_id = "sell", llm_reasoning, held["id"]
            else:
                reason = llm_reasoning or "llm rejected exit"
        else:
            reason = "not_a_candidate" if held is None else "score_above_exit_threshold_or_unavailable"

        evaluation_row = models.log_opportunity_evaluation(
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
            trade_id=trade_id,
        )
        if calibration_to_log is not None:
            models.log_confidence_calibration(
                opportunity_evaluation_id=evaluation_row["id"], **calibration_to_log
            )

    process_closed_trades(mode)

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
