"""Step 9: real rolling multi-fold walk-forward validation — genuinely new,
distinct from src/learning/simulation.py's existing single-split logic
(which only re-scores trades ALREADY taken by live/paper trading; this
runs the actual BacktestEngine over held-out historical windows the
strategy never saw). N rolling folds, non-overlapping, stepped forward by
the test window's length: train window -> test window (never touched
during train) -> evaluate -> next window."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import mean, stdev

from src.backtest.engine import BacktestEngine
from src.backtest.performance_analyzer import analyze
from src.config import (
    BACKTEST_WALK_FORWARD_N_FOLDS,
    BACKTEST_WALK_FORWARD_TEST_DAYS,
    BACKTEST_WALK_FORWARD_TRAIN_DAYS,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
    SIGNIFICANCE_THRESHOLD,
)
from src.learning.statistics import z_test_two_means


@dataclass
class Fold:
    fold_number: int
    train_window_start: date
    train_window_end: date
    test_window_start: date
    test_window_end: date
    in_sample_metrics: dict | None
    out_of_sample_metrics: dict | None
    p_value: float | None
    passed: bool | None


def _run_window(symbols, symbol_to_pair, start, end, params_json, engine_kwargs) -> tuple[list, dict]:
    engine = BacktestEngine(symbols, symbol_to_pair, start, end, params_json=params_json, **engine_kwargs)
    result = engine.run()
    metrics = analyze(result["closed_trades"], result["snapshots"], engine.portfolio.starting_capital)
    return result["closed_trades"], metrics


def run_walk_forward(
    symbols: list[str],
    symbol_to_pair: dict[str, str],
    overall_start: date,
    overall_end: date,
    params_json: dict,
    n_folds: int = BACKTEST_WALK_FORWARD_N_FOLDS,
    train_days: int = BACKTEST_WALK_FORWARD_TRAIN_DAYS,
    test_days: int = BACKTEST_WALK_FORWARD_TEST_DAYS,
    engine_kwargs: dict | None = None,
) -> list[Fold]:
    """params_json is applied UNCHANGED to both train and test windows in
    each fold — this validates whether a fixed parameter set (e.g. an
    adaptive_strategy_versions candidate) holds up out-of-sample across
    multiple historical periods, not a per-fold parameter search (a caller
    wanting to compare candidates re-runs this once per candidate and
    hands the results to strategy_comparison.py)."""
    engine_kwargs = engine_kwargs or {}
    folds: list[Fold] = []
    window_start = overall_start
    for fold_number in range(1, n_folds + 1):
        train_start = window_start
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > overall_end:
            break

        train_trades, in_sample = _run_window(symbols, symbol_to_pair, train_start, train_end, params_json, engine_kwargs)
        test_trades, out_of_sample = _run_window(symbols, symbol_to_pair, test_start, test_end, params_json, engine_kwargs)

        p_value = None
        passed = None
        n_train, n_test = in_sample["trades_count"], out_of_sample["trades_count"]
        # Below RECOMMENDATION_MIN_SAMPLE_SIZE per side: reported as
        # "insufficient sample" (None), never a fabricated p-value — same
        # policy as everywhere else in this codebase, and the deliberate
        # substitution for a parametric Student's-t test (see config.py's
        # Statistical methods note).
        if n_train >= RECOMMENDATION_MIN_SAMPLE_SIZE and n_test >= RECOMMENDATION_MIN_SAMPLE_SIZE:
            train_pnls = [t.pnl for t in train_trades]
            test_pnls = [t.pnl for t in test_trades]
            p_value = z_test_two_means(
                mean(train_pnls), stdev(train_pnls), n_train, mean(test_pnls), stdev(test_pnls), n_test
            )
            train_expectancy = in_sample.get("expectancy")
            test_expectancy = out_of_sample.get("expectancy")
            if p_value is not None and train_expectancy is not None and test_expectancy is not None:
                # Fails only on a STATISTICALLY SIGNIFICANT degradation —
                # no significant train/test difference (stable) or test
                # doing as-well-or-better both count as passing.
                passed = p_value >= SIGNIFICANCE_THRESHOLD or test_expectancy >= train_expectancy

        folds.append(
            Fold(
                fold_number=fold_number,
                train_window_start=train_start,
                train_window_end=train_end,
                test_window_start=test_start,
                test_window_end=test_end,
                in_sample_metrics=in_sample,
                out_of_sample_metrics=out_of_sample,
                p_value=p_value,
                passed=passed,
            )
        )
        window_start = window_start + timedelta(days=test_days)

    return folds
