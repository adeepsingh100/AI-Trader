"""Walk-forward validation + simulation before adoption (Steps 9-10).

Never validates against future data: a recommendation is regenerated
using only the OLDER (train) fraction of LEARNING_HISTORY_WINDOW_DAYS,
then tested only against the NEWER (test) fraction — never touched during
generation. Overall evidence gate is status.can_simulate() (Evidence-
Driven Learning Progression, Stage 3) — stricter than the hypothesis-stage
floor by design, since preventing look-ahead bias means both halves must
*also* independently clear RECOMMENDATION_MIN_SAMPLE_SIZE on top of that.

"Simulation" here means exactly what recommendations.py already
documents: re-scoring/re-partitioning trades that were ALREADY TAKEN,
never inventing a counterfactual trade that wasn't. On a statistically
significant pass (two-sample z-test, SIGNIFICANCE_THRESHOLD) that ALSO
clears status.can_create_candidate() (Stage 4 — a simulation can pass at
Stage 3 without yet being allowed to create a candidate), a
strategy_simulations row is written and an
adaptive_strategy_versions candidate is created LAZILY — only for
proposals that clear both bars, so that table only ever holds genuine,
sufficiently-evidenced candidates. A failing simulation, or one that
passes below the candidate-creation floor, still writes its
strategy_simulations row (fully auditable — Step 15's "rejected
recommendations" report, and the research_note explains which case it
was) but no version candidate.

Scientific Strategy Optimization Framework additions: a bootstrap
confidence-interval gate (backtest.statistical_validation, previously
orphaned) runs alongside the z-test for threshold/exit-params candidates —
skipped for weight candidates, whose validation is about win/loss
SEPARATION, not a specific trade subset's returns, so a returns-CI doesn't
apply there. Exit-params candidates additionally get a real BacktestEngine
replay (backtest.strategy_comparison + walk_forward_validator, both
previously orphaned) when historical candle data exists AND a caller
supplies symbol_to_pair (the one thing that needs a network-derived
value) — gracefully skipped, never a crash, when either is unavailable."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from statistics import stdev

from src.backtest.statistical_validation import bootstrap_confidence_interval
from src.config import (
    ADAPTIVE_TRAIN_TEST_SPLIT_PCT,
    BACKTEST_TICK_TIMEFRAME,
    LEARNING_HISTORY_WINDOW_DAYS,
    LEARNING_STAGE_VALIDATION_MIN_TRADES,
    MIN_OPPORTUNITY_SCORE,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
    SIGNIFICANCE_THRESHOLD,
)
from src.db import models
from src.learning.feature_importance import compute_subscore_correlation_weights, score_separation_p_value
from src.learning.fitness import compute_fitness_score
from src.learning.learning_status import LearningStatus, compute_learning_status
from src.learning.recommendations import _simulate_exit_pnl, current_weights
from src.learning.statistics import compute_bucket_statistics, z_test_two_means
from src.utils import parse_timestamp as _parse_ts


def _fetch_trades(mode: str, strategy_type: str = "default") -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    return [t for t in models.get_recently_closed_trades(mode, since, strategy_type) if t.get("pnl") is not None]


def _train_test_split(trades: list[dict], split_pct: float = ADAPTIVE_TRAIN_TEST_SPLIT_PCT):
    ordered = sorted(trades, key=lambda t: t["closed_at"])
    split_index = int(len(ordered) * split_pct)
    return ordered[:split_index], ordered[split_index:]


def _bootstrap_gate(returns: list[float]) -> dict:
    """Bootstrap CI on a candidate's test-window returns — an additional,
    distributional-assumption-free confidence check alongside the z-test.
    Cleared only if the CI's LOWER bound is still positive, not just the
    point estimate — a candidate whose apparent edge could plausibly be
    zero or negative doesn't pass on a single lucky sample."""
    ci = bootstrap_confidence_interval(returns)
    return {"ci": ci, "cleared": ci is not None and ci["ci_low"] > 0}


def _backtest_replay_gate(
    symbols: list[str], symbol_to_pair: dict[str, str], start: date, end: date,
    baseline_params: dict, candidate_params: dict,
) -> dict | None:
    """One-window backtest replay, baseline vs candidate params over the
    same historical period — feeds both into strategy_comparison.compare,
    reused as-is (statistically-gated pairwise A/B, not "whichever raw
    number is bigger")."""
    from src.backtest.engine import BacktestEngine
    from src.backtest.performance_analyzer import analyze
    from src.backtest.strategy_comparison import compare

    baseline_engine = BacktestEngine(symbols, symbol_to_pair, start, end, params_json=baseline_params)
    baseline_result = baseline_engine.run()
    baseline_metrics = analyze(
        baseline_result["closed_trades"], baseline_result["snapshots"], baseline_engine.portfolio.starting_capital
    )

    candidate_engine = BacktestEngine(symbols, symbol_to_pair, start, end, params_json=candidate_params)
    candidate_result = candidate_engine.run()
    candidate_metrics = analyze(
        candidate_result["closed_trades"], candidate_result["snapshots"], candidate_engine.portfolio.starting_capital
    )

    return compare(baseline_result["closed_trades"], candidate_result["closed_trades"], baseline_metrics, candidate_metrics)


def _walk_forward_gate(symbols: list[str], symbol_to_pair: dict[str, str], start: date, end: date, candidate_params: dict):
    from src.backtest.walk_forward_validator import run_walk_forward

    return run_walk_forward(symbols, symbol_to_pair, start, end, candidate_params)


def _has_historical_candles(symbols: list[str], symbol_to_pair: dict[str, str], start: date, end: date) -> bool:
    if not symbols or not symbol_to_pair.get(symbols[0]):
        return False
    start_ms = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    return models.historical_candles_exist(symbol_to_pair[symbols[0]], BACKTEST_TICK_TIMEFRAME, start_ms, end_ms)


def _build_research_note(
    param_name: str,
    candidate_value,
    baseline_stats: dict,
    candidate_stats: dict,
    hypothesis_rationale: str | None,
    weakness: dict | None,
    walk_forward_summary: str | None,
    passed: bool,
    stage_gate_note: str | None = None,
) -> str:
    """Observation/Weakness/Hypothesis/Simulation/Walk Forward/Decision —
    a narrative research report, not a changelog line (Step 11). `passed`
    here means "a candidate was actually created" (Progressive Learning
    Stages: a statistically-passing simulation below
    LEARNING_STAGE_VALIDATION_MIN_TRADES is deferred, not promoted — the
    caller passes candidate_created, not the raw statistical pass/fail, so
    this note never claims "Promoted" for a candidate that doesn't exist)."""
    lines = [
        f"Observation: baseline {param_name} performance over "
        f"{baseline_stats.get('trades_count') or 0} trades, "
        f"profit factor {baseline_stats.get('profit_factor')}, expectancy {baseline_stats.get('expectancy')}."
    ]
    if weakness:
        lines.append(f"Weakness: {weakness}")
    if hypothesis_rationale:
        lines.append(f"Hypothesis: {hypothesis_rationale}")
    lines.append(
        f"Simulation: candidate {param_name}={candidate_value} shows profit factor "
        f"{candidate_stats.get('profit_factor')}, expectancy {candidate_stats.get('expectancy')} "
        f"over {candidate_stats.get('trades_count') or 0} trades."
    )
    if walk_forward_summary:
        lines.append(f"Walk Forward: {walk_forward_summary}")
    if stage_gate_note:
        lines.append(f"Stage gate: {stage_gate_note}")
    lines.append(f"Decision: {'Promoted' if passed else 'Rejected'}")
    return "\n".join(lines)


def _create_candidate_version(
    mode: str,
    batch_id: str | None,
    simulation_id: int,
    params_json: dict,
    status: LearningStatus,
    fitness_score: float | None = None,
    strategy_type: str = "default",
) -> dict | None:
    """Progressive Learning Stages, Stage 4: a candidate row is only ever
    created once status.can_create_candidate() clears — a statistically-
    passing simulation below that floor is real (its strategy_simulations
    row still records passed=True) but candidate creation is deferred, not
    skipped forever; the caller re-simulates every run and creates the row
    as soon as enough evidence accumulates."""
    if not status.can_create_candidate():
        return None
    latest = models.get_latest_adaptive_strategy_version(mode, strategy_type=strategy_type)
    next_version_number = (latest["version_number"] + 1) if latest else 1
    return models.insert_adaptive_strategy_version(
        mode=mode,
        version_number=next_version_number,
        params_json=params_json,
        source_recommendation_batch_id=batch_id,
        source_simulation_id=simulation_id,
        notes="Auto-generated candidate from a passing walk-forward simulation.",
        fitness_score=fitness_score,
        strategy_type=strategy_type,
    )


def simulate_weight_recommendation(
    mode: str, batch_id: str | None = None, status: LearningStatus | None = None, strategy_type: str = "default"
) -> dict | None:
    """Re-derives a candidate weight set using only the TRAIN window, then
    tests whether it separates winners from losers on the TEST window
    better than the current live weights do on that same out-of-sample
    window. None if there isn't enough trade volume to independently
    clear the sample floor on both halves."""
    status = status or compute_learning_status(mode, strategy_type)
    if not status.can_simulate():
        return None

    all_trades = _fetch_trades(mode, strategy_type)

    train, test = _train_test_split(all_trades)
    if len(train) < RECOMMENDATION_MIN_SAMPLE_SIZE or len(test) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return None

    candidate_weights = compute_subscore_correlation_weights(mode, trades=train, cache=False, strategy_type=strategy_type)
    if candidate_weights is None:
        return None

    candidate_metrics = score_separation_p_value(test, candidate_weights)
    baseline_metrics = score_separation_p_value(test, current_weights())
    if candidate_metrics is None:
        return None

    baseline_p = baseline_metrics["p_value"] if baseline_metrics else None
    passed = (
        candidate_metrics["p_value"] is not None
        and candidate_metrics["p_value"] < SIGNIFICANCE_THRESHOLD
        and (baseline_p is None or candidate_metrics["p_value"] < baseline_p)
    )

    simulation_row = models.insert_strategy_simulation(
        recommendation_batch_id=batch_id,
        mode=mode,
        train_window_start=_parse_ts(train[0]["closed_at"]),
        train_window_end=_parse_ts(train[-1]["closed_at"]),
        test_window_start=_parse_ts(test[0]["closed_at"]),
        test_window_end=_parse_ts(test[-1]["closed_at"]),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        p_value=candidate_metrics["p_value"],
        passed=passed,
        strategy_type=strategy_type,
    )

    if passed:
        _create_candidate_version(
            mode, batch_id, simulation_row["id"], candidate_weights, status=status, strategy_type=strategy_type
        )

    return simulation_row


def simulate_threshold_recommendation(
    mode: str, status: LearningStatus | None = None, strategy_type: str = "default"
) -> dict | None:
    """Walk-forward validation for the MIN_OPPORTUNITY_SCORE threshold
    recommendation specifically — the one threshold metric with a single,
    stable score extractor (opportunity_score). Per-symbol optimal_*
    metrics (recommendations.generate_symbol_recommendations) aren't
    walk-forward-simulated here: they already need RECOMMENDATION_MIN_SAMPLE_SIZE
    trades per symbol just to be generated once, and a second per-symbol
    train/test split would need that count to roughly double again —
    revisit once real trade volume makes it worth the added complexity."""
    latest = models.get_latest_recommendation(mode, "MIN_OPPORTUNITY_SCORE", strategy_type=strategy_type)
    if latest is None or latest.get("recommended_value") is None or latest.get("status") != "pending":
        return None

    status = status or compute_learning_status(mode, strategy_type)
    if not status.can_simulate():
        return None

    all_trades = _fetch_trades(mode, strategy_type)
    train, test = _train_test_split(all_trades)
    if len(train) < RECOMMENDATION_MIN_SAMPLE_SIZE or len(test) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return None

    candidate_threshold = latest["recommended_value"]
    batch_id = latest.get("batch_id")
    capital_config = models.get_capital_config(mode, strategy_type)
    capital_to_use = capital_config["capital_to_use"] if capital_config else 0

    scored_test = []
    for trade in test:
        entry_eval = models.get_entry_evaluation_for_trade(trade["id"])
        if entry_eval and entry_eval.get("opportunity_score") is not None:
            scored_test.append((entry_eval["opportunity_score"], trade))

    baseline_trades = [t for score, t in scored_test if score >= MIN_OPPORTUNITY_SCORE]
    candidate_trades = [t for score, t in scored_test if score >= candidate_threshold]
    baseline_stats = compute_bucket_statistics(baseline_trades, capital_to_use)
    candidate_stats = compute_bucket_statistics(candidate_trades, capital_to_use)

    baseline_returns = [t["pnl"] / capital_to_use for t in baseline_trades] if capital_to_use else []
    candidate_returns = [t["pnl"] / capital_to_use for t in candidate_trades] if capital_to_use else []

    p_value, passed = None, False
    if (
        len(baseline_returns) >= 2
        and len(candidate_returns) >= 2
        and candidate_stats["expectancy"] is not None
        and baseline_stats["expectancy"] is not None
    ):
        p_value = z_test_two_means(
            sum(candidate_returns) / len(candidate_returns),
            stdev(candidate_returns),
            len(candidate_returns),
            sum(baseline_returns) / len(baseline_returns),
            stdev(baseline_returns),
            len(baseline_returns),
        )
        passed = (
            p_value is not None
            and p_value < SIGNIFICANCE_THRESHOLD
            and candidate_stats["expectancy"] > baseline_stats["expectancy"]
        )

    validation_detail = {}
    if passed:
        gate = _bootstrap_gate(candidate_returns)
        validation_detail["bootstrap_ci"] = gate["ci"]
        passed = passed and gate["cleared"]

    candidate_created = False
    stage_gate_note = None
    if passed:
        if not status.can_create_candidate():
            stage_gate_note = (
                f"statistically valid but deferred — needs {LEARNING_STAGE_VALIDATION_MIN_TRADES} "
                f"total closed trades (have {status.trades_collected})."
            )
        else:
            candidate_created = True

    research_note = _build_research_note(
        "MIN_OPPORTUNITY_SCORE",
        candidate_threshold,
        baseline_stats,
        candidate_stats,
        latest.get("rationale"),
        None,
        None,
        candidate_created,
        stage_gate_note=stage_gate_note,
    )

    simulation_row = models.insert_strategy_simulation(
        recommendation_batch_id=batch_id,
        mode=mode,
        train_window_start=_parse_ts(train[0]["closed_at"]),
        train_window_end=_parse_ts(train[-1]["closed_at"]),
        test_window_start=_parse_ts(test[0]["closed_at"]),
        test_window_end=_parse_ts(test[-1]["closed_at"]),
        baseline_metrics=baseline_stats,
        candidate_metrics=candidate_stats,
        p_value=p_value,
        passed=passed,
        research_note=research_note,
        validation_detail=validation_detail or None,
        strategy_type=strategy_type,
    )

    if candidate_created:
        fitness = compute_fitness_score(candidate_stats, capital_to_use)
        _create_candidate_version(
            mode,
            batch_id,
            simulation_row["id"],
            {"MIN_OPPORTUNITY_SCORE": candidate_threshold},
            status=status,
            fitness_score=fitness["fitness_score"],
            strategy_type=strategy_type,
        )

    return simulation_row


def _activate_exit_params_candidate(candidate: dict, strategy_type: str = "default") -> dict | None:
    """Auto-promotes an exit-params candidate straight into a new active
    strategy_versions row — the only candidate type auto-activated today
    (weight/regime/symbol candidates target OPPORTUNITY_WEIGHT_*-style env
    vars, not a DB row, so there's no equivalent safe automated path for
    those without a redeploy). Real money is unaffected by this step
    alone: the new version only reaches real trading after independently
    clearing src.learning.promotion_gate.evaluate_promotion() on ITS OWN
    paper trade history — get_closed_trades(mode, version["id"]) scopes to
    the new version's id, so its sample-size/paper-days clocks start at
    zero."""
    current = models.get_latest_version(strategy_type)
    if current is None:
        return None
    current_params = current.get("params_json") or {}
    merged_params = {**current_params, **candidate["params_json"]}
    new_version = models.insert_strategy_version(
        version_number=current["version_number"] + 1,
        prompt_text=current["prompt_text"],
        params_json=merged_params,
        notes=(
            f"Auto-activated from adaptive_strategy_versions candidate "
            f"{candidate['id']} (fitness={candidate.get('fitness_score')})."
        ),
        strategy_type=strategy_type,
    )
    models.log_agent_event(
        "adaptive_strategy_engine",
        "info",
        f"AUTO-ACTIVATED strategy_versions id={new_version['id']} "
        f"version_number={new_version['version_number']} from candidate {candidate['id']}",
    )
    return new_version


def simulate_exit_params_recommendation(
    mode: str,
    symbol_to_pair: dict[str, str] | None = None,
    status: LearningStatus | None = None,
    strategy_type: str = "default",
) -> list[dict]:
    """Walk-forward validation for stop_loss_pct/take_profit_pct candidates
    (recommendations.generate_exit_params_recommendations) — the new lever
    the retired evolution_agent.propose_next_version used to guess at
    freely. Always runs the always-available re-partition z-test +
    bootstrap CI check (via recommendations._simulate_exit_pnl's mfe_pct/
    mae_pct approximation); additionally runs a real BacktestEngine replay
    (§6-7: strategy_comparison + walk_forward_validator) when historical
    candle data exists for the version's traded symbols AND symbol_to_pair
    is supplied — the nightly caller builds this from a live markets_details
    fetch, the one thing here that needs network data; test/CLI callers
    that omit it get the always-available check only, never a crash."""
    status = status or compute_learning_status(mode, strategy_type)
    if not status.can_simulate():
        return []

    results = []
    for param_name in ("stop_loss_pct", "take_profit_pct"):
        latest = models.get_latest_recommendation(mode, param_name, strategy_type=strategy_type)
        if latest is None or latest.get("recommended_value") is None or latest.get("status") != "pending":
            continue

        all_trades = _fetch_trades(mode, strategy_type)
        train, test = _train_test_split(all_trades)
        if len(train) < RECOMMENDATION_MIN_SAMPLE_SIZE or len(test) < RECOMMENDATION_MIN_SAMPLE_SIZE:
            continue

        candidate_value = latest["recommended_value"]
        batch_id = latest.get("batch_id")
        capital_config = models.get_capital_config(mode, strategy_type)
        capital_to_use = capital_config["capital_to_use"] if capital_config else 0

        version = models.get_latest_version(strategy_type)
        current_params = (version.get("params_json") or {}) if version else {}
        other_leg = "take_profit_pct" if param_name == "stop_loss_pct" else "stop_loss_pct"
        fixed_other = current_params.get(other_leg)
        stop = candidate_value if param_name == "stop_loss_pct" else fixed_other
        target = candidate_value if param_name == "take_profit_pct" else fixed_other
        candidate_trades = [{**t, "pnl": _simulate_exit_pnl(t, stop, target)} for t in test]

        baseline_stats = compute_bucket_statistics(test, capital_to_use)
        candidate_stats = compute_bucket_statistics(candidate_trades, capital_to_use)
        if baseline_stats["expectancy"] is None or candidate_stats["expectancy"] is None:
            continue

        baseline_returns = [t["pnl"] / capital_to_use for t in test] if capital_to_use else []
        candidate_returns = [t["pnl"] / capital_to_use for t in candidate_trades] if capital_to_use else []

        p_value, passed = None, False
        if len(baseline_returns) >= 2 and len(candidate_returns) >= 2:
            p_value = z_test_two_means(
                sum(candidate_returns) / len(candidate_returns),
                stdev(candidate_returns),
                len(candidate_returns),
                sum(baseline_returns) / len(baseline_returns),
                stdev(baseline_returns),
                len(baseline_returns),
            )
            passed = (
                p_value is not None
                and p_value < SIGNIFICANCE_THRESHOLD
                and candidate_stats["expectancy"] > baseline_stats["expectancy"]
            )

        validation_detail = {}
        if passed:
            gate = _bootstrap_gate(candidate_returns)
            validation_detail["bootstrap_ci"] = gate["ci"]
            passed = passed and gate["cleared"]

        walk_forward_summary = None
        if passed and symbol_to_pair:
            symbols = sorted({t["symbol"] for t in test})
            start = _parse_ts(test[0]["closed_at"]).date()
            end = _parse_ts(test[-1]["closed_at"]).date()
            if _has_historical_candles(symbols, symbol_to_pair, start, end):
                candidate_params = {**current_params, param_name: candidate_value}
                comparison = _backtest_replay_gate(symbols, symbol_to_pair, start, end, current_params, candidate_params)
                if comparison is not None:
                    validation_detail["strategy_comparison"] = comparison
                    if comparison.get("winner") == "a":
                        passed = False
                    walk_forward_summary = (
                        f"backtest replay winner={comparison.get('winner')}, "
                        f"p_values={comparison.get('p_values')}"
                    )
                if passed:
                    folds = _walk_forward_gate(symbols, symbol_to_pair, start, end, candidate_params)
                    if folds:
                        validation_detail["walk_forward_folds"] = [
                            {"fold_number": f.fold_number, "p_value": f.p_value, "passed": f.passed} for f in folds
                        ]
                        if any(f.passed is False for f in folds):
                            passed = False
                        fold_summary = f"{sum(1 for f in folds if f.passed)}/{len(folds)} folds passed"
                        walk_forward_summary = (
                            f"{walk_forward_summary}; {fold_summary}" if walk_forward_summary else fold_summary
                        )

        candidate_created = False
        stage_gate_note = None
        if passed:
            if not status.can_create_candidate():
                stage_gate_note = (
                    f"statistically valid but deferred — needs {LEARNING_STAGE_VALIDATION_MIN_TRADES} "
                    f"total closed trades (have {status.trades_collected})."
                )
            else:
                candidate_created = True

        research_note = _build_research_note(
            param_name, candidate_value, baseline_stats, candidate_stats,
            latest.get("rationale"), None, walk_forward_summary, candidate_created,
            stage_gate_note=stage_gate_note,
        )

        simulation_row = models.insert_strategy_simulation(
            recommendation_batch_id=batch_id,
            mode=mode,
            train_window_start=_parse_ts(train[0]["closed_at"]),
            train_window_end=_parse_ts(train[-1]["closed_at"]),
            test_window_start=_parse_ts(test[0]["closed_at"]),
            test_window_end=_parse_ts(test[-1]["closed_at"]),
            baseline_metrics=baseline_stats,
            candidate_metrics=candidate_stats,
            p_value=p_value,
            passed=passed,
            research_note=research_note,
            validation_detail=validation_detail or None,
            strategy_type=strategy_type,
        )

        if candidate_created:
            fitness = compute_fitness_score(candidate_stats, capital_to_use)
            candidate_row = _create_candidate_version(
                mode, batch_id, simulation_row["id"], {param_name: candidate_value},
                status=status, fitness_score=fitness["fitness_score"], strategy_type=strategy_type,
            )
            if candidate_row is not None:
                _activate_exit_params_candidate(candidate_row, strategy_type)

        results.append(simulation_row)

    return results
