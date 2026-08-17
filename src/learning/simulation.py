"""Walk-forward validation + simulation before adoption (Steps 9-10).

Never validates against future data: a recommendation is regenerated
using only the OLDER (train) fraction of LEARNING_HISTORY_WINDOW_DAYS,
then tested only against the NEWER (test) fraction — never touched during
generation. Both halves must independently clear
RECOMMENDATION_MIN_SAMPLE_SIZE, so this realistically stays silent until
there's roughly double the trade volume generate_recommendations() itself
needs — that's the honest cost of preventing look-ahead bias, not a bug.

"Simulation" here means exactly what recommendations.py already
documents: re-scoring/re-partitioning trades that were ALREADY TAKEN,
never inventing a counterfactual trade that wasn't. On a statistically
significant pass (two-sample z-test, SIGNIFICANCE_THRESHOLD), a
strategy_simulations row is written and an adaptive_strategy_versions
candidate is created LAZILY — only for proposals that clear the bar, so
that table only ever holds genuine candidates. A failing simulation still
writes its strategy_simulations row (fully auditable — Step 15's
"rejected recommendations" report) but no version candidate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import stdev

from src.config import (
    ADAPTIVE_TRAIN_TEST_SPLIT_PCT,
    LEARNING_HISTORY_WINDOW_DAYS,
    MIN_OPPORTUNITY_SCORE,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
    SIGNIFICANCE_THRESHOLD,
)
from src.db import models
from src.learning.feature_importance import compute_subscore_correlation_weights, score_separation_p_value
from src.learning.recommendations import current_weights
from src.learning.statistics import compute_bucket_statistics, z_test_two_means
from src.utils import parse_timestamp as _parse_ts


def _fetch_trades(mode: str) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    return [t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None]


def _train_test_split(trades: list[dict], split_pct: float = ADAPTIVE_TRAIN_TEST_SPLIT_PCT):
    ordered = sorted(trades, key=lambda t: t["closed_at"])
    split_index = int(len(ordered) * split_pct)
    return ordered[:split_index], ordered[split_index:]


def _create_candidate_version(mode: str, batch_id: str | None, simulation_id: int, params_json: dict) -> dict:
    latest = models.get_latest_adaptive_strategy_version(mode)
    next_version_number = (latest["version_number"] + 1) if latest else 1
    return models.insert_adaptive_strategy_version(
        mode=mode,
        version_number=next_version_number,
        params_json=params_json,
        source_recommendation_batch_id=batch_id,
        source_simulation_id=simulation_id,
        notes=(
            "Auto-generated candidate from a passing walk-forward simulation. "
            "Not active — requires explicit human approval (status='approved') "
            "before any manual promotion into live config."
        ),
    )


def simulate_weight_recommendation(mode: str, batch_id: str | None = None) -> dict | None:
    """Re-derives a candidate weight set using only the TRAIN window, then
    tests whether it separates winners from losers on the TEST window
    better than the current live weights do on that same out-of-sample
    window. None if there isn't enough trade volume to independently
    clear the sample floor on both halves."""
    all_trades = _fetch_trades(mode)
    if len(all_trades) < RECOMMENDATION_MIN_SAMPLE_SIZE * 2:
        return None

    train, test = _train_test_split(all_trades)
    if len(train) < RECOMMENDATION_MIN_SAMPLE_SIZE or len(test) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return None

    candidate_weights = compute_subscore_correlation_weights(mode, trades=train, cache=False)
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
    )

    if passed:
        _create_candidate_version(mode, batch_id, simulation_row["id"], candidate_weights)

    return simulation_row


def simulate_threshold_recommendation(mode: str) -> dict | None:
    """Walk-forward validation for the MIN_OPPORTUNITY_SCORE threshold
    recommendation specifically — the one threshold metric with a single,
    stable score extractor (opportunity_score). Per-symbol optimal_*
    metrics (recommendations.generate_symbol_recommendations) aren't
    walk-forward-simulated here: they already need RECOMMENDATION_MIN_SAMPLE_SIZE
    trades per symbol just to be generated once, and a second per-symbol
    train/test split would need that count to roughly double again —
    revisit once real trade volume makes it worth the added complexity."""
    latest = models.get_latest_recommendation(mode, "MIN_OPPORTUNITY_SCORE")
    if latest is None or latest.get("recommended_value") is None or latest.get("status") != "pending":
        return None

    all_trades = _fetch_trades(mode)
    if len(all_trades) < RECOMMENDATION_MIN_SAMPLE_SIZE * 2:
        return None
    train, test = _train_test_split(all_trades)
    if len(train) < RECOMMENDATION_MIN_SAMPLE_SIZE or len(test) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return None

    candidate_threshold = latest["recommended_value"]
    batch_id = latest.get("batch_id")
    capital_config = models.get_capital_config(mode)
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
    )

    if passed:
        _create_candidate_version(
            mode, batch_id, simulation_row["id"], {"MIN_OPPORTUNITY_SCORE": candidate_threshold}
        )

    return simulation_row
