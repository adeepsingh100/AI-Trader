"""One full cycle: scan -> feature engine -> opportunity score -> top
candidates -> risk -> execute -> log. Single invocation, runs to
completion, exits — see PROJECT_SPEC.md §5.

Fully quant, zero LLM calls anywhere in this module: OpportunityScorer
(src/features/) deterministically scores every scanned symbol, and
reaching MIN_OPPORTUNITY_SCORE/TOP_N_CANDIDATES (entry) or dropping below
EXIT_SCORE_THRESHOLD (exit) is necessary but not sufficient — an entry
candidate must ALSO clear the Net Expectancy Gate (risk_manager.
compute_net_expectancy_pct: fees/GST/TDS/spread/slippage netted against
the resolved stop/target and the system's own calibrated win-probability
estimate) before an order is placed. Both gates are pure code, no LLM, no
second opinion asked of anything. An LLM does get consulted elsewhere in
this codebase, but only once an hour, offline from live trading, to
propose stop_loss_pct/take_profit_pct candidates that still have to clear
the same statistical gate as every other strategy change (src/learning/
recommendations.py::generate_ai_exit_params_recommendations) — a bad or
unavailable LLM call there costs one skipped hourly proposal, never a
blocked trade.

Spot-only: an accepted exit closes an existing held position for that
symbol (there's no shorting on CoinDCX spot). An accepted entry opens one
if the Risk Manager clears it. Circuit breaker is checked before every
buy, not just once at cycle start, since a loss realized earlier in the
same cycle must stop later buys in that same cycle.

Stop-loss/take-profit (from the active version's params_json) is swept
every cycle too — see run_risk_check() below, which does only that sweep,
meant for a tighter cron than the full cycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.agents.data_agent import get_market_snapshot
from src.agents.execution.paper import PaperExecutionAgent
from src.agents.execution.real import RealExecutionAgent
from src.agents.risk_manager import (
    circuit_breaker_triggered,
    committed_capital,
    compute_net_expectancy_pct,
    evaluate,
    exit_reason,
    resolve_exit_params,
    target_hit,
    today_ist,
)
from src.audit.trail import config_version
from src.coindcx_client import get_ticker
from src.config import (
    BUCKET_MODIFIER_CAP,
    BUCKET_MODIFIER_SENSITIVITY,
    LEARNING_CATCHUP_LOOKBACK_HOURS,
    MIN_FINAL_CONFIDENCE,
    PAPER_TRADES_ON_NEGATIVE_EXPECTANCY,
    RECENT_PERFORMANCE_LOOKBACK_TRADES,
    RECENT_STREAK_LOSS_MODIFIER_CAP,
    RECENT_STREAK_WIN_MODIFIER_CAP,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
    STRATEGY_PROFILES,
)
from src.db import models
from src.features.feature_engine import compute_multi_timeframe_features
from src.features.opportunity_scorer import (
    primary_timeframe,
    score_opportunity,
    select_top_candidates,
)
from src.learning.confidence_calibration import calibrate_confidence
from src.learning.statistics import process_closed_trades, streaks
from src.learning.trade_memory import find_similar_trades
from src.monitoring.metrics import log_resource_snapshot, track
from src.portfolio.intelligence import Position, correlation, returns_series

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


def _recent_performance_modifier(mode: str, strategy_type: str) -> float | None:
    """Adaptive confidence chain (Step 7): current win/loss streak over
    the last RECENT_PERFORMANCE_LOOKBACK_TRADES closed trades, scaled to
    the streak length and capped — deliberately asymmetric (a losing
    streak can suppress confidence more than a winning streak inflates
    it). Computed once per cycle, not per candidate. Scoped to this
    strategy_type's own trades — a swing streak shouldn't move default's
    confidence or vice versa."""
    since = datetime.now(timezone.utc) - timedelta(hours=LEARNING_CATCHUP_LOOKBACK_HOURS)
    recent = sorted(
        (t for t in models.get_recently_closed_trades(mode, since, strategy_type) if t.get("pnl") is not None),
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


def _price_history_from_snapshot(scored: list[dict]) -> dict[str, list[float]]:
    """Daily closes for every symbol scanned this cycle — already-fetched
    data (market["candles_by_timeframe"]["1d"]), no extra API calls.
    Symbols not in this cycle's top-N snapshot (a held position that's
    fallen out of top turnover) simply aren't keys here; Portfolio
    Intelligence degrades correlation/beta for them to None rather than
    guessing — concentration/sector checks don't need price history at
    all, so they're unaffected."""
    history: dict[str, list[float]] = {}
    for r in scored:
        daily = r["market"]["candles_by_timeframe"].get("1d")
        if daily:
            history[r["symbol"]] = [c["close"] for c in daily if c.get("close") is not None]
    return history


def _portfolio_positions(open_trades: list[dict], last_prices: dict[str, float]) -> list[Position]:
    """Current price falls back to entry_price for a held symbol this
    cycle's snapshot didn't cover — an approximation (not a fresh ticker
    price), but keeps concentration math running rather than skipping a
    position it does hold."""
    return [
        Position(t["symbol"], t["qty"], t["entry_price"], last_prices.get(t["symbol"], t["entry_price"]))
        for t in open_trades
    ]


def _avg_correlation_with_book(symbol: str, open_trades: list[dict], price_history: dict[str, list[float]]) -> float | None:
    candidate_prices = price_history.get(symbol)
    if not candidate_prices:
        return None
    candidate_returns = returns_series(candidate_prices)
    correlations = []
    for t in open_trades:
        held_prices = price_history.get(t["symbol"])
        if not held_prices or t["symbol"] == symbol:
            continue
        c = correlation(candidate_returns, returns_series(held_prices))
        if c is not None:
            correlations.append(c)
    return sum(correlations) / len(correlations) if correlations else None


def _sweep_stop_loss_take_profit(
    mode: str,
    strategy_type: str,
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
        reason = exit_reason(
            trade["entry_price"],
            price,
            params_json,
            stored_stop_loss_price=trade.get("stop_loss_price"),
            stored_take_profit_price=trade.get("take_profit_price"),
        )
        if reason is None:
            continue

        fill = execution_agent.place_order(trade["symbol"], "sell", trade["qty"], price)
        _, daily_pnl = _record_close(
            mode, strategy_type, capital_config, trade, fill, daily_pnl, exit_reason_value=reason
        )
        models.log_agent_event("orchestrator", "info", f"{reason} exit {trade['symbol']}")
        closed.append(trade)
        open_trades.remove(trade)
        del open_by_symbol[trade["symbol"]]

    return closed, daily_pnl


def _opportunity_summary(
    record: dict,
    held: dict | None = None,
    historical_context: dict | None = None,
    timeframe_weights: dict[str, float] | None = None,
) -> dict:
    """Curated digest for the LLM — never the raw multi-timeframe feature
    dump (160+ floats), which would strain the token budget for no gain
    over what a human/LLM actually needs to judge the call."""
    primary = record["features_by_tf"].get(primary_timeframe(timeframe_weights)) or {}
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
    mode: str,
    strategy_type: str,
    capital_config: dict,
    trade: dict,
    fill: dict,
    daily_pnl: dict,
    exit_reason_value: str | None = None,
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
        strategy_type=strategy_type,
    )
    return pnl, updated


def run_cycle(mode: str = MODE, execution_agent=None, n_symbols: int = 10) -> dict:
    """Loops every strategy_type with a seeded capital_config row for this
    mode (STRATEGY_PROFILES intersected with models.get_active_strategy_types
    — a type with no capital_config never runs, the "ships dormant" activation
    gate), running each one's own independent scoring/entry/exit pass. The
    market snapshot (candle fetch) and per-symbol feature computation are
    fetched/computed ONCE and shared across every active type's scoring pass
    — only the weighting/thresholds differ per type, not the raw candles."""
    if execution_agent is None:
        execution_agent = PaperExecutionAgent() if mode == "paper" else RealExecutionAgent()

    active_types = [t for t in models.get_active_strategy_types(mode) if t in STRATEGY_PROFILES]
    if not active_types:
        if mode != "real":
            # Real mode legitimately sits unconfigured for a long time
            # (below); paper mode with NO strategy_type configured at all
            # means seed_config.py never ran — a real setup error, still
            # worth a loud crash rather than a silent no-op forever.
            raise RuntimeError(f"no capital_config row for mode={mode!r} — insert one first")
        return {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "no_capital_config"}

    # Phase 1: per-type preflight (capital_config/paused/version/daily_pnl,
    # circuit breaker, and the stop-loss/take-profit sweep) — none of this
    # needs the market snapshot. A type that's paused/unconfigured/tripped
    # is fully resolved here without ever fetching candles for it.
    results = {"opened": [], "closed": [], "circuit_breaker": False, "by_strategy_type": {}}
    ready_states: dict[str, dict] = {}
    for strategy_type in active_types:
        try:
            outcome = _preflight_strategy_type(mode, strategy_type, execution_agent)
        except Exception as e:
            # Same per-unit fault isolation as _process_candidate below,
            # one level up: one strategy_type's setup error must not take
            # down every other active type's cycle.
            models.log_agent_event(
                "orchestrator", "error", f"strategy_type={strategy_type}: {type(e).__name__}: {e}"
            )
            continue
        if outcome.get("_ready"):
            ready_states[strategy_type] = outcome
            continue
        results["by_strategy_type"][strategy_type] = outcome
        results["opened"].extend(outcome["opened"])
        results["closed"].extend(outcome["closed"])
        results["circuit_breaker"] = results["circuit_breaker"] or outcome["circuit_breaker"]

    # Phase 2: only fetched/computed if at least one type actually needs it —
    # every type paused/unconfigured/already-tripped this cycle means zero
    # wasted API calls, same as the original single-strategy behavior.
    if ready_states:
        with track("data_agent", "market_snapshot"):
            snapshot = get_market_snapshot(n_symbols)
        features_by_symbol = {
            market["symbol"]: (market, compute_multi_timeframe_features(market["candles_by_timeframe"]))
            for market in snapshot
        }
        for strategy_type, state in ready_states.items():
            try:
                r = _score_and_process_strategy_type(mode, strategy_type, execution_agent, features_by_symbol, state)
            except Exception as e:
                models.log_agent_event(
                    "orchestrator", "error", f"strategy_type={strategy_type}: {type(e).__name__}: {e}"
                )
                continue
            results["by_strategy_type"][strategy_type] = r
            results["opened"].extend(r["opened"])
            results["closed"].extend(r["closed"])
            results["circuit_breaker"] = results["circuit_breaker"] or r["circuit_breaker"]

    log_resource_snapshot("orchestrator")
    return results


def _preflight_strategy_type(mode: str, strategy_type: str, execution_agent) -> dict:
    """Everything about a strategy_type's cycle that doesn't need the
    market snapshot: capital_config/pause/version resolution, daily_pnl,
    the circuit breaker, and the stop-loss/take-profit sweep. Returns a
    terminal result dict (opened/closed/circuit_breaker, cycle already
    fully handled for this type) OR a `{"_ready": True, ...}` state dict
    for run_cycle to hand to _score_and_process_strategy_type once the
    shared snapshot is fetched."""
    capital_config = models.get_capital_config(mode, strategy_type)

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
        version = models.get_latest_promoted_version(strategy_type)
        if version is None:
            models.log_agent_event(
                "orchestrator", "info",
                f"real mode ({strategy_type}): no promoted strategy version yet, skipping",
            )
            return {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "no_promoted_version"}
    else:
        if capital_config is None:
            raise RuntimeError(f"no capital_config row for mode={mode!r} strategy_type={strategy_type!r}")
        if capital_config.get("paused"):
            return {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "paused"}
        version = models.get_latest_version(strategy_type)
        if version is None:
            # Two different situations share "no active version": never
            # bootstrapped at all for this strategy_type (seed_config.py
            # never ran — a real setup error, still worth a loud crash) vs.
            # every existing version is suspended (Strategy Health Engine,
            # PROJECT_SPEC.md §3d — a legitimate, reversible outcome that
            # must no-op, not crash-loop the cron every 10 minutes).
            if not models.get_all_strategy_versions():
                raise RuntimeError("no strategy_versions row — create one first")
            models.log_agent_event(
                "orchestrator", "info",
                f"paper mode ({strategy_type}): every strategy version is suspended, skipping",
            )
            return {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "no_active_version"}

    daily_pnl = models.get_daily_pnl(today_ist(), mode, strategy_type) or _empty_daily_pnl()

    if circuit_breaker_triggered(daily_pnl, capital_config):
        execution_agent.flatten_all(mode, strategy_type)
        return {"opened": [], "closed": [], "circuit_breaker": True}

    open_trades = models.get_open_trades(mode, strategy_type)
    open_by_symbol = {t["symbol"]: t for t in open_trades}

    opened, closed = [], []
    stopped, daily_pnl = _sweep_stop_loss_take_profit(
        mode, strategy_type, capital_config, version, open_trades, open_by_symbol, daily_pnl, execution_agent
    )
    closed.extend(stopped)

    if circuit_breaker_triggered(daily_pnl, capital_config):
        execution_agent.flatten_all(mode, strategy_type)
        return {"opened": opened, "closed": closed, "circuit_breaker": True}

    return {
        "_ready": True,
        "capital_config": capital_config,
        "version": version,
        "daily_pnl": daily_pnl,
        "open_trades": open_trades,
        "open_by_symbol": open_by_symbol,
        "opened": opened,
        "closed": closed,
    }


def _score_and_process_strategy_type(
    mode: str, strategy_type: str, execution_agent, features_by_symbol: dict, state: dict
) -> dict:
    profile = STRATEGY_PROFILES[strategy_type]
    capital_config = state["capital_config"]
    version = state["version"]
    daily_pnl = state["daily_pnl"]
    open_trades = state["open_trades"]
    open_by_symbol = state["open_by_symbol"]
    opened = state["opened"]
    closed = state["closed"]

    # Pass 1: pure, no LLM, no side effects — score every scanned symbol
    # with THIS strategy_type's own weight/timeframe profile. Candles and
    # per-timeframe features (features_by_symbol) were already fetched/
    # computed once, shared across every active type this cycle — only
    # the weighting pass below is per-type.
    scored = []
    for symbol, (market, features_by_tf) in features_by_symbol.items():
        scores = score_opportunity(features_by_tf, profile["opportunity_weights"], profile["timeframe_weights"])
        scored.append({"symbol": symbol, "market": market, "features_by_tf": features_by_tf, **scores})

    not_held = [r for r in scored if r["symbol"] not in open_by_symbol]
    candidate_symbols = {
        r["symbol"]
        for r in select_top_candidates(
            not_held, top_n=profile["top_n_candidates"], min_score=profile["min_opportunity_score"]
        )
    }

    # Adaptive confidence chain (Step 7) inputs — fetched once per cycle,
    # not once per candidate (Step 13: cached/batched, not redundant
    # per-candidate queries). overall_win_rate uses the active version's
    # own learning_statistics bucket (already computed, already cached)
    # as the baseline regime/symbol win rates are compared against,
    # rather than a fresh full-trades scan every cycle. Scoped to this
    # strategy_type so its stats never blend with another type's.
    regime_stats = {
        r["dimension_value"]: r
        for r in models.get_learning_statistics(mode, dimension_type="market_regime", strategy_type=strategy_type)
    }
    symbol_stats = {
        r["dimension_value"]: r
        for r in models.get_learning_statistics(mode, dimension_type="symbol", strategy_type=strategy_type)
    }
    version_stats = {
        r["dimension_value"]: r
        for r in models.get_learning_statistics(
            mode, dimension_type="strategy_version", strategy_type=strategy_type
        )
    }
    overall_row = version_stats.get(str(version["id"]))
    overall_win_rate = overall_row.get("win_rate") if overall_row else None
    recent_performance_modifier = _recent_performance_modifier(mode, strategy_type)

    # Portfolio Intelligence + Capital Allocation inputs (PROJECT_SPEC.md
    # §3d) — built once per cycle from data already fetched this cycle
    # (no extra API calls), reused by every candidate's risk_manager.evaluate
    # call below.
    price_history = _price_history_from_snapshot(scored)
    last_price_by_symbol = {r["symbol"]: r["market"]["last_price"] for r in scored}

    # Pass 2: LLM validation only for entry candidates and held positions
    # whose score has deteriorated past the exit threshold — everyone else
    # is logged with no LLM call at all.
    #
    # Per-symbol fault isolation (Resilience, PROJECT_SPEC.md §3d): the body
    # is a nested function so one symbol's exception (a malformed LLM
    # response, a transient execution error) can be caught and logged by
    # the loop below without aborting every remaining symbol in the cycle —
    # a confirmed real gap before this. `nonlocal daily_pnl` is needed
    # because the exit branch reassigns it (via _record_close) and that
    # update must be visible to the next iteration's circuit-breaker check.
    def _process_candidate(record: dict) -> None:
        nonlocal daily_pnl

        symbol = record["symbol"]
        market = record["market"]
        held = open_by_symbol.get(symbol)
        opportunity_score = record["opportunity_score"]

        llm_decision = llm_reasoning = llm_raw_response = risk_manager_result = None
        final_decision, reason = "hold", None
        trade_id = None
        calibration_to_log = None

        if held is None and symbol in candidate_symbols:
            similar = find_similar_trades(
                record, market_regime=record["market_regime"], mode=mode, strategy_type=strategy_type
            )
            summary = _opportunity_summary(
                record, historical_context=similar, timeframe_weights=profile["timeframe_weights"]
            )
            # Reaching this branch already means the quant scorer accepted
            # the candidate (this profile's min_opportunity_score/top_n) —
            # no separate validation step, no LLM call.
            llm_decision = "accept"
            llm_reasoning = (
                f"quant score {opportunity_score:.1f} >= {profile['min_opportunity_score']} "
                f"(trend={record['trend_score']:.0f} momentum={record['momentum_score']:.0f} "
                f"volume={record['volume_score']:.0f} volatility={record['volatility_score']:.0f} "
                f"resistance_headroom={record['risk_score']:.0f}, regime={record['market_regime']})"
            )
            llm_raw_response = summary

            historical_confidence = similar["win_rate"] * 100 if similar["win_rate"] is not None else None
            regime_modifier = _bucket_modifier(regime_stats, record["market_regime"], overall_win_rate)
            symbol_modifier = _bucket_modifier(symbol_stats, symbol, overall_win_rate)
            calibrated = calibrate_confidence(
                opportunity_score,
                historical_confidence,
                similar["count"],
                regime_modifier=regime_modifier,
                symbol_modifier=symbol_modifier,
                recent_performance_modifier=recent_performance_modifier,
            )
            calibration_to_log = {
                "ai_confidence": opportunity_score,
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

            if confidence_cleared:
                capital_to_use = capital_config.get("capital_to_use") or 0
                current_exposure_pct = (
                    committed_capital(open_trades) / capital_to_use * 100 if capital_to_use else None
                )

                # Net Expectancy Gate: a real risk boundary (evidence-
                # validated params_json stop/target, falling back to an
                # ATR-derived one only for a leg params_json doesn't
                # configure — resolve_exit_params) and the fees/GST/TDS/
                # spread/slippage this trade would actually incur decide
                # whether it's worth taking — opportunity_score/confidence
                # clearing their own gates is necessary but not sufficient.
                atr_pct = (
                    record["features_by_tf"].get(primary_timeframe(profile["timeframe_weights"])) or {}
                ).get("atr_pct")
                params_json = version.get("params_json") or {}
                stop_loss_pct, take_profit_pct = resolve_exit_params(
                    params_json,
                    atr_pct,
                    profile["stop_loss_atr_multiplier"],
                    profile["take_profit_atr_multiplier"],
                    profile["exit_param_sweep_min_pct"],
                    profile["exit_param_sweep_max_pct"],
                )
                win_probability_pct = (
                    calibrated["final_confidence"]
                    if calibrated["final_confidence"] is not None
                    else opportunity_score
                )
                net_expectancy = compute_net_expectancy_pct(
                    stop_loss_pct, take_profit_pct, win_probability_pct / 100
                )
                expectancy_cleared = net_expectancy is not None and (
                    net_expectancy["net_expectancy_pct"] > 0
                    or (mode == "paper" and PAPER_TRADES_ON_NEGATIVE_EXPECTANCY)
                )
                ne_display = f"{net_expectancy['net_expectancy_pct']:.4f}" if net_expectancy is not None else "n/a"
                llm_reasoning += (
                    f"; stop={stop_loss_pct} target={take_profit_pct} atr_pct={atr_pct} "
                    f"exposure_pct={current_exposure_pct} net_expectancy_pct={ne_display}"
                )

                if expectancy_cleared:
                    sizing_context = {
                        "avg_correlation": _avg_correlation_with_book(symbol, open_trades, price_history),
                        "candidate_volatility_pct": atr_pct,
                        # Not computed live yet — no live mark-to-market equity
                        # curve exists to derive it from (that's backtest-only,
                        # src/backtest/portfolio_manager.py); the drawdown
                        # factor defaults to neutral (1.0) rather than a
                        # fabricated estimate.
                        "recent_drawdown_pct": None,
                        "current_exposure_pct": current_exposure_pct,
                        "strategy_win_rate": overall_win_rate,
                        "market_regime": record["market_regime"],
                        "confidence": calibrated["final_confidence"],
                    }
                    decision = evaluate(
                        capital_config,
                        daily_pnl,
                        open_trades,
                        market["last_price"],
                        symbol=symbol,
                        portfolio_positions=_portfolio_positions(open_trades, last_price_by_symbol),
                        price_history=price_history,
                        sizing_context=sizing_context,
                        stop_loss_pct=stop_loss_pct,
                        risk_per_trade_pct=profile["risk_per_trade_pct"],
                    )
                    risk_manager_result = decision.action
                    if decision.action == "size":
                        fill = execution_agent.place_order(symbol, "buy", decision.qty, market["last_price"])
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
                else:
                    reason = f"net_expectancy gated: {ne_display}"
            else:
                reason = f"confidence gated: {calibrated['final_confidence']:.1f} < {MIN_FINAL_CONFIDENCE}"

        elif (
            held is not None
            and opportunity_score is not None
            and opportunity_score < profile["exit_score_threshold"]
        ):
            # Score dropping below this profile's exit_score_threshold is
            # itself the exit decision — no separate validation step, no
            # LLM call.
            llm_decision = "accept"
            llm_reasoning = f"quant score {opportunity_score:.1f} < {profile['exit_score_threshold']} exit threshold"
            llm_raw_response = None

            fill = execution_agent.place_order(symbol, "sell", held["qty"], market["last_price"])
            _, daily_pnl = _record_close(
                mode, strategy_type, capital_config, held, fill, daily_pnl, exit_reason_value="ai_exit"
            )
            closed.append(held)
            open_trades.remove(held)
            del open_by_symbol[symbol]
            final_decision, reason, trade_id = "sell", llm_reasoning, held["id"]
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
            market_regime=record["market_regime"],
            config_version=config_version(),
        )
        if calibration_to_log is not None:
            models.log_confidence_calibration(
                opportunity_evaluation_id=evaluation_row["id"], **calibration_to_log
            )

    for record in scored:
        if circuit_breaker_triggered(daily_pnl, capital_config):
            execution_agent.flatten_all(mode, strategy_type)
            break
        try:
            _process_candidate(record)
        except Exception as e:
            models.log_agent_event(
                "orchestrator", "error", f"{record['symbol']}: {type(e).__name__}: {e}"
            )

    process_closed_trades(mode, strategy_type)

    return {
        "opened": opened,
        "closed": closed,
        "circuit_breaker": circuit_breaker_triggered(daily_pnl, capital_config),
    }


def run_risk_check(mode: str = MODE, execution_agent=None) -> dict:
    """No LLM calls, no market snapshot — just the circuit breaker and
    the stop-loss/take-profit sweep, per active strategy_type. Meant to
    run on a tighter cron than run_cycle (which is throttled by LLM
    budget), so a bad move gets cut off sooner than a full 10-minute
    wait."""
    if execution_agent is None:
        execution_agent = PaperExecutionAgent() if mode == "paper" else RealExecutionAgent()

    active_types = [t for t in models.get_active_strategy_types(mode) if t in STRATEGY_PROFILES]
    if not active_types:
        return {"closed": [], "circuit_breaker": False, "skipped": "no_capital_config"}

    results = {"closed": [], "circuit_breaker": False, "by_strategy_type": {}}
    for strategy_type in active_types:
        r = _run_risk_check_for_strategy_type(mode, strategy_type, execution_agent)
        results["by_strategy_type"][strategy_type] = r
        results["closed"].extend(r["closed"])
        results["circuit_breaker"] = results["circuit_breaker"] or r["circuit_breaker"]
    return results


def _run_risk_check_for_strategy_type(mode: str, strategy_type: str, execution_agent) -> dict:
    capital_config = models.get_capital_config(mode, strategy_type)
    if capital_config is None or capital_config.get("paused"):
        return {"closed": [], "circuit_breaker": False, "skipped": "not_configured_or_paused"}

    version = (
        models.get_latest_promoted_version(strategy_type)
        if mode == "real"
        else models.get_latest_version(strategy_type)
    )
    if version is None:
        return {"closed": [], "circuit_breaker": False, "skipped": "no_version"}

    daily_pnl = models.get_daily_pnl(today_ist(), mode, strategy_type) or _empty_daily_pnl()

    if circuit_breaker_triggered(daily_pnl, capital_config):
        execution_agent.flatten_all(mode, strategy_type)
        return {"closed": [], "circuit_breaker": True}

    open_trades = models.get_open_trades(mode, strategy_type)
    open_by_symbol = {t["symbol"]: t for t in open_trades}
    closed, daily_pnl = _sweep_stop_loss_take_profit(
        mode, strategy_type, capital_config, version, open_trades, open_by_symbol, daily_pnl, execution_agent
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
