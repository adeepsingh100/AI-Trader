"""BacktestEngine — the event reactor. Not a reuse of orchestrator.run_cycle
(that's a live-polling shell: Supabase writes and get_ticker() every call,
fundamentally incompatible with "replay chronologically, no network calls
in the hot loop"). What IS reused, unchanged: feature_engine, opportunity_
scorer, and risk_manager's pure functions — that's where "same trading
logic in both modes" is actually honored. Replicates all THREE circuit-
breaker checkpoints run_cycle has (top-of-decision-pass, post-SL/TP-sweep,
per-candidate) — dropping any one changes economic behavior.

Two cadences, kept deliberately separate from the SimulationClock's tick
granularity: BACKTEST_DECISION_CYCLE_MINUTES (scoring+risk+entry/exit,
mirrors trading_cycle.yml's */10) and BACKTEST_RISK_CHECK_MINUTES
(stop-loss/take-profit sweep only, mirrors risk_check.yml's */5). The tick
timeframe (default 1m) governs candle visibility ONLY — firing the
decision pass on every tick would simulate a bot checking 5-10x more often
than the live one ever does.

Symbol universe is explicit (the `symbols` argument), never reconstructed
from a live turnover ranking — CoinDCX has no historical ticker/turnover
series to replay, so defaulting to "today's top-N over history" would be
survivorship bias. Every report built from a run should say this."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from src.agents.risk_manager import (
    RiskDecision,
    circuit_breaker_triggered,
    evaluate,
    exit_reason,
)
from src.backtest.data_provider import CandleStore
from src.backtest.execution_simulator import execute_market_order
from src.backtest.order_manager import OrderManager
from src.backtest.portfolio_manager import PortfolioManager
from src.config import (
    BACKTEST_DECISION_CYCLE_MINUTES,
    BACKTEST_MAX_CONCURRENT_POSITIONS,
    BACKTEST_POSITION_SIZE_PCT,
    BACKTEST_RISK_CHECK_MINUTES,
    BACKTEST_STARTING_CAPITAL,
    BACKTEST_TICK_TIMEFRAME,
    BACKTEST_WARMUP_BUFFER_DAYS,
    EXIT_SCORE_THRESHOLD,
    FEATURE_CANDLE_LIMIT,
    FEATURE_TIMEFRAMES,
    PORTFOLIO_CORRELATION_LOOKBACK_BARS,
)
from src.db import models
from src.features.feature_engine import compute_multi_timeframe_features
from src.features.opportunity_scorer import score_opportunity, select_top_candidates
from src.portfolio.intelligence import Position as IntelligencePosition

from src.backtest.simulation_clock import SimulationClock

SURVIVORSHIP_BIAS_NOTE = (
    "Symbol universe is a fixed, user-supplied list — CoinDCX has no "
    "historical ticker/turnover series to replay, so this run cannot "
    "reconstruct what would have ranked top-N by live turnover on any "
    "given historical date. Results only speak to these exact symbols."
)


def _date_to_ms(d: date, end_of_day: bool = False) -> int:
    t = datetime.combine(d, time.max if end_of_day else time.min, tzinfo=timezone.utc)
    return int(t.timestamp() * 1000)


class BacktestEngine:
    def __init__(
        self,
        symbols: list[str],
        symbol_to_pair: dict[str, str],
        start_date: date,
        end_date: date,
        params_json: dict | None = None,
        starting_capital: float = BACKTEST_STARTING_CAPITAL,
        position_size_pct: float = BACKTEST_POSITION_SIZE_PCT,
        max_concurrent_positions: int = BACKTEST_MAX_CONCURRENT_POSITIONS,
        daily_profit_target: float = float("inf"),
        max_daily_loss: float = float("inf"),
        warmup_buffer_days: int = BACKTEST_WARMUP_BUFFER_DAYS,
        tick_timeframe: str = BACKTEST_TICK_TIMEFRAME,
        decision_cycle_minutes: int = BACKTEST_DECISION_CYCLE_MINUTES,
        risk_check_minutes: int = BACKTEST_RISK_CHECK_MINUTES,
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.params_json = params_json or {}
        self.capital_config = {
            "capital_to_use": starting_capital,
            "position_size_pct": position_size_pct,
            "max_concurrent_positions": max_concurrent_positions,
            "daily_profit_target": daily_profit_target,
            "max_daily_loss": max_daily_loss,
        }
        self.tick_timeframe = tick_timeframe
        self.decision_cycle_ms = decision_cycle_minutes * 60_000
        self.risk_check_ms = risk_check_minutes * 60_000

        run_start_ms = _date_to_ms(start_date)
        run_end_ms = _date_to_ms(end_date, end_of_day=True)
        warmup_start_ms = _date_to_ms(start_date - timedelta(days=warmup_buffer_days))

        # CandleStores load [warmup_start, end] so lookback works from day 1
        # of the requested window; the clock itself only ticks the
        # requested [start, end] — nothing (decision cycle, risk check,
        # circuit breaker) ever fires during the warm-up period itself.
        self.stores: dict[str, dict[str, CandleStore]] = {
            symbol: {
                tf: CandleStore(symbol_to_pair[symbol], tf, warmup_start_ms, run_end_ms)
                for tf in FEATURE_TIMEFRAMES
            }
            for symbol in symbols
        }
        self.clock = SimulationClock(run_start_ms, run_end_ms, tick_timeframe)

        self.portfolio = PortfolioManager(starting_capital)
        self.order_manager = OrderManager()
        self.execution_history: list[dict] = []
        self.daily_pnl = {"realized_pnl": 0.0, "trades_count": 0, "circuit_breaker_triggered": False}
        self._current_day = self.clock.today_ist()
        self._next_decision_ms = run_start_ms
        self._next_risk_check_ms = run_start_ms
        self.circuit_breaker_events = 0

    # --- internals ---

    def _reference_prices(self, as_of_ms: int) -> dict[str, float]:
        prices = {}
        for symbol, tf_stores in self.stores.items():
            price = tf_stores[self.tick_timeframe].current_bar_open_price(as_of_ms)
            if price is not None:
                prices[symbol] = price
        return prices

    def _bar_volume(self, symbol: str, as_of_ms: int) -> float:
        candles = self.stores[symbol][self.tick_timeframe].visible_slice(as_of_ms + 1, 1)
        return float(candles[-1]["volume"]) if candles else 0.0

    def _score_symbol(self, symbol: str, as_of_ms: int, price: float) -> dict | None:
        candles_by_tf = {
            tf: self.stores[symbol][tf].visible_slice(as_of_ms, FEATURE_CANDLE_LIMIT) for tf in FEATURE_TIMEFRAMES
        }
        features_by_tf = compute_multi_timeframe_features(candles_by_tf)
        scores = score_opportunity(features_by_tf)
        return {"symbol": symbol, "last_price": price, "features_by_tf": features_by_tf, **scores}

    def _open_trades_as_dicts(self) -> list[dict]:
        return [{"qty": p.qty, "entry_price": p.entry_price} for p in self.portfolio.positions.values()]

    def _price_history(self, as_of_ms: int) -> dict[str, list[float]]:
        """Daily closes per symbol, visible as of as_of_ms — mirrors
        orchestrator._price_history_from_snapshot's daily-close convention.
        Already look-ahead-safe: CandleStore.visible_slice only returns bars
        closed as of as_of_ms, same guarantee portfolio/intelligence.py
        requires of its caller."""
        history: dict[str, list[float]] = {}
        for symbol, tf_stores in self.stores.items():
            daily_store = tf_stores.get("1d")
            if daily_store is None:
                continue
            candles = daily_store.visible_slice(as_of_ms, PORTFOLIO_CORRELATION_LOOKBACK_BARS)
            closes = [c["close"] for c in candles if c.get("close") is not None]
            if closes:
                history[symbol] = closes
        return history

    def _portfolio_positions_for_intelligence(self, prices: dict[str, float]) -> list[IntelligencePosition]:
        return [
            IntelligencePosition(p.symbol, p.qty, p.entry_price, prices.get(p.symbol, p.entry_price))
            for p in self.portfolio.positions.values()
        ]

    def _flatten_all(self, time_ms: int, prices: dict[str, float]) -> None:
        now = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
        for symbol in list(self.portfolio.positions.keys()):
            price = prices.get(symbol)
            if price is None:
                continue
            pos = self.portfolio.positions[symbol]
            fill = execute_market_order(symbol, "sell", pos.qty, price, self._bar_volume(symbol, time_ms))
            self._log_execution(now, fill, "market")
            if fill.status == "rejected":
                continue
            trade = self.portfolio.close_position(
                symbol, fill.fill_price, now, fill.fees, fill.slippage_cost, exit_reason="circuit_breaker"
            )
            self.daily_pnl["realized_pnl"] += trade.pnl
            self.daily_pnl["trades_count"] += 1
        self.daily_pnl["circuit_breaker_triggered"] = True
        self.circuit_breaker_events += 1

    def _log_execution(self, time_: datetime, fill, order_type: str) -> None:
        self.execution_history.append(
            {
                "symbol": fill.symbol,
                "order_type": order_type,
                "side": fill.side,
                "requested_qty": fill.qty,
                "requested_price": fill.fill_price,
                "status": fill.status,
                "filled_qty": fill.qty if fill.status != "rejected" else 0.0,
                "filled_price": fill.fill_price if fill.status != "rejected" else None,
                "rejection_reason": fill.rejection_reason,
                "event_time": time_.isoformat(),
            }
        )

    def _maybe_roll_day(self) -> None:
        today = self.clock.today_ist()
        if today != self._current_day:
            self._current_day = today
            self.daily_pnl = {"realized_pnl": 0.0, "trades_count": 0, "circuit_breaker_triggered": False}

    def _risk_check_pass(self, time_ms: int, prices: dict[str, float]) -> None:
        """Mirrors orchestrator._sweep_stop_loss_take_profit + its
        bracketing circuit-breaker checks exactly (checkpoint 1 of 3)."""
        if circuit_breaker_triggered(self.daily_pnl, self.capital_config):
            self._flatten_all(time_ms, prices)
            return
        now = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
        for symbol in list(self.portfolio.positions.keys()):
            price = prices.get(symbol)
            if price is None:
                continue
            self.portfolio.update_excursion(symbol, price)
            pos = self.portfolio.positions[symbol]
            reason = exit_reason(pos.entry_price, price, self.params_json)
            if reason is None:
                continue
            fill = execute_market_order(symbol, "sell", pos.qty, price, self._bar_volume(symbol, time_ms))
            self._log_execution(now, fill, "market")
            if fill.status == "rejected":
                continue
            trade = self.portfolio.close_position(symbol, fill.fill_price, now, fill.fees, fill.slippage_cost, exit_reason=reason)
            self.daily_pnl["realized_pnl"] += trade.pnl
            self.daily_pnl["trades_count"] += 1
            self.daily_pnl["circuit_breaker_triggered"] = circuit_breaker_triggered(self.daily_pnl, self.capital_config)

        if circuit_breaker_triggered(self.daily_pnl, self.capital_config):
            self._flatten_all(time_ms, prices)

    def _decision_pass(self, time_ms: int, prices: dict[str, float]) -> None:
        """Mirrors orchestrator.run_cycle's Pass 1 + Pass 2 (checkpoints 2
        and 3 of 3: top-of-pass and per-candidate)."""
        if circuit_breaker_triggered(self.daily_pnl, self.capital_config):
            self._flatten_all(time_ms, prices)
            return

        now = datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc)
        scored = []
        for symbol, price in prices.items():
            record = self._score_symbol(symbol, time_ms, price)
            if record is not None:
                scored.append(record)

        not_held = [r for r in scored if r["symbol"] not in self.portfolio.positions]
        candidate_symbols = {r["symbol"] for r in select_top_candidates(not_held)}
        price_history = self._price_history(time_ms)

        for record in scored:
            if circuit_breaker_triggered(self.daily_pnl, self.capital_config):
                self._flatten_all(time_ms, prices)
                break

            symbol = record["symbol"]
            held = self.portfolio.positions.get(symbol)
            opportunity_score = record["opportunity_score"]

            if held is None and symbol in candidate_symbols:
                # MIN_OPPORTUNITY_SCORE/TOP_N_CANDIDATES clearance (already
                # true — select_top_candidates filtered on it) is the entry
                # decision, deterministic — mirrors orchestrator.py.
                confidence = opportunity_score
                decision: RiskDecision = evaluate(
                    self.capital_config,
                    self.daily_pnl,
                    self._open_trades_as_dicts(),
                    record["last_price"],
                    symbol=symbol,
                    portfolio_positions=self._portfolio_positions_for_intelligence(prices),
                    price_history=price_history,
                )
                if decision.action != "size":
                    continue
                fill = execute_market_order(
                    symbol, "buy", decision.qty, record["last_price"], self._bar_volume(symbol, time_ms)
                )
                self._log_execution(now, fill, "market")
                if fill.status == "rejected":
                    continue
                stop_loss_pct = self.params_json.get("stop_loss_pct")
                take_profit_pct = self.params_json.get("take_profit_pct")
                self.portfolio.open_position(
                    symbol,
                    fill.qty,
                    fill.fill_price,
                    now,
                    fill.fees,
                    stop_loss_price=fill.fill_price * (1 - stop_loss_pct) if stop_loss_pct else None,
                    take_profit_price=fill.fill_price * (1 + take_profit_pct) if take_profit_pct else None,
                    confidence=confidence,
                    opportunity_score=opportunity_score,
                    market_regime=record["market_regime"],
                )

            elif held is not None and opportunity_score is not None and opportunity_score < EXIT_SCORE_THRESHOLD:
                # Score dropping below EXIT_SCORE_THRESHOLD is itself the
                # exit decision — mirrors orchestrator.py.
                fill = execute_market_order(
                    symbol, "sell", held.qty, record["last_price"], self._bar_volume(symbol, time_ms)
                )
                self._log_execution(now, fill, "market")
                if fill.status == "rejected":
                    continue
                trade = self.portfolio.close_position(
                    symbol, fill.fill_price, now, fill.fees, fill.slippage_cost, exit_reason="ai_exit"
                )
                self.daily_pnl["realized_pnl"] += trade.pnl
                self.daily_pnl["trades_count"] += 1

    # --- entry point ---

    def run(self) -> dict:
        for t in self.clock.ticks():
            self._maybe_roll_day()
            prices = self._reference_prices(t)
            for symbol, price in prices.items():
                self.portfolio.update_excursion(symbol, price)

            if t >= self._next_risk_check_ms:
                self._next_risk_check_ms = t + self.risk_check_ms
                self._risk_check_pass(t, prices)

            if t >= self._next_decision_ms:
                self._next_decision_ms = t + self.decision_cycle_ms
                self._decision_pass(t, prices)
                now = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
                self.portfolio.snapshot(now, prices)

        return {
            "closed_trades": self.portfolio.closed_trades,
            "open_positions": list(self.portfolio.positions.values()),
            "snapshots": self.portfolio.snapshots,
            "execution_history": self.execution_history,
            "circuit_breaker_events": self.circuit_breaker_events,
            "survivorship_bias_note": SURVIVORSHIP_BIAS_NOTE,
        }


def run_and_persist(
    symbols: list[str],
    symbol_to_pair: dict[str, str],
    start_date: date,
    end_date: date,
    params_json: dict | None = None,
    name: str | None = None,
    **engine_kwargs,
) -> int:
    """Runs a backtest end to end and persists the run/trades/snapshots/
    execution history/performance metrics — the CLI entry point's
    implementation, also reusable directly (e.g. by walk_forward_validator
    callers that want a persisted run per fold). Returns the new run_id."""
    from src.backtest.performance_analyzer import analyze
    from src.backtest.trade_analysis import to_rows

    engine = BacktestEngine(symbols, symbol_to_pair, start_date, end_date, params_json=params_json, **engine_kwargs)
    run_row = models.insert_backtest_run(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        warmup_buffer_days=engine_kwargs.get("warmup_buffer_days", BACKTEST_WARMUP_BUFFER_DAYS),
        starting_capital=engine.capital_config["capital_to_use"],
        params_json=params_json or {},
        name=name,
    )
    run_id = run_row["id"]
    try:
        result = engine.run()
    except Exception:
        models.update_backtest_run_status(run_id, "failed")
        raise

    for row in to_rows(result["closed_trades"]):
        models.insert_backtest_trade(run_id, row)
    if result["snapshots"]:
        models.insert_backtest_portfolio_snapshots(
            run_id,
            [{**s, "snapshot_time": s["snapshot_time"].isoformat()} for s in result["snapshots"]],
        )
    if result["execution_history"]:
        models.insert_backtest_execution_events(run_id, result["execution_history"])

    metrics = analyze(result["closed_trades"], result["snapshots"], engine.portfolio.starting_capital)
    models.insert_backtest_performance_metrics(run_id, metrics)
    models.update_backtest_run_status(run_id, "completed", completed_at=datetime.now(timezone.utc))

    return run_id


if __name__ == "__main__":
    import argparse
    import json as _json

    from src.coindcx_client import get_markets_details, symbol_to_pair as _symbol_to_pair

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True, help="comma-separated, e.g. BTCINR,ETHINR")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--params-file", help="JSON file with stop_loss_pct/take_profit_pct/etc")
    parser.add_argument("--name")
    cli_args = parser.parse_args()

    cli_symbols = [s.strip() for s in cli_args.symbols.split(",") if s.strip()]
    cli_details = get_markets_details()
    cli_symbol_to_pair = {s: _symbol_to_pair(s, cli_details) for s in cli_symbols}
    cli_params = _json.load(open(cli_args.params_file)) if cli_args.params_file else {}

    new_run_id = run_and_persist(
        cli_symbols,
        cli_symbol_to_pair,
        date.fromisoformat(cli_args.start),
        date.fromisoformat(cli_args.end),
        params_json=cli_params,
        name=cli_args.name,
    )
    print(f"backtest_runs.id={new_run_id}")
    print(SURVIVORSHIP_BIAS_NOTE)
