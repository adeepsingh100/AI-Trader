"""Advisory threshold recommendations — never auto-applied to config,
human approval required. Compares the current MIN_OPPORTUNITY_SCORE's
realized expectancy against higher score bands; writes a recommendation
only when a higher band clearly outperforms with enough samples, and only
when that differs materially from the latest existing recommendation
(idempotent by construction — otherwise this would write a near-duplicate
row every night forever). Nightly (evolution_agent's cron), not per-cycle."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from src.config import (
    LEARNING_HISTORY_WINDOW_DAYS,
    MIN_OPPORTUNITY_SCORE,
    OPPORTUNITY_SCORE_BUCKET_WIDTH,
    RECOMMENDATION_MIN_IMPROVEMENT_PCT,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
)
from src.db import models
from src.learning.statistics import compute_bucket_statistics


def generate_recommendations(mode: str) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    closed = [t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None]
    if len(closed) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return []

    scored_trades = []
    for trade in closed:
        entry_eval = models.get_entry_evaluation_for_trade(trade["id"])
        if entry_eval and entry_eval.get("opportunity_score") is not None:
            scored_trades.append((entry_eval["opportunity_score"], trade))
    if len(scored_trades) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return []

    capital_config = models.get_capital_config(mode)
    capital_to_use = capital_config["capital_to_use"] if capital_config else 0

    baseline_trades = [t for score, t in scored_trades if score >= MIN_OPPORTUNITY_SCORE]
    baseline_stats = compute_bucket_statistics(baseline_trades, capital_to_use)
    if baseline_stats["expectancy"] is None:
        return []

    candidate_thresholds = sorted(
        {
            math.floor(score / OPPORTUNITY_SCORE_BUCKET_WIDTH) * OPPORTUNITY_SCORE_BUCKET_WIDTH
            for score, _ in scored_trades
            if score > MIN_OPPORTUNITY_SCORE
        }
    )

    best_threshold, best_expectancy, best_sample_size = None, baseline_stats["expectancy"], 0
    for threshold in candidate_thresholds:
        higher_trades = [t for score, t in scored_trades if score >= threshold]
        if len(higher_trades) < RECOMMENDATION_MIN_SAMPLE_SIZE:
            continue
        stats = compute_bucket_statistics(higher_trades, capital_to_use)
        if stats["expectancy"] is not None and stats["expectancy"] > best_expectancy:
            best_threshold, best_expectancy, best_sample_size = threshold, stats["expectancy"], len(higher_trades)

    if best_threshold is None:
        return []

    if baseline_stats["expectancy"]:
        improvement_pct = (best_expectancy - baseline_stats["expectancy"]) / abs(baseline_stats["expectancy"]) * 100
    else:
        improvement_pct = float("inf") if best_expectancy > 0 else 0.0

    if improvement_pct < RECOMMENDATION_MIN_IMPROVEMENT_PCT:
        return []

    latest = models.get_latest_recommendation(mode, "MIN_OPPORTUNITY_SCORE")
    if latest is not None:
        prior = latest["recommended_value"] or 1.0
        moved_pct = abs(latest["recommended_value"] - best_threshold) / abs(prior) * 100
        if moved_pct < RECOMMENDATION_MIN_IMPROVEMENT_PCT:
            return []  # not materially different from what's already on record

    rationale = (
        f"Trades scoring >= {best_threshold} (n={best_sample_size}) show expectancy "
        f"{best_expectancy:.2f} vs {baseline_stats['expectancy']:.2f} at the current "
        f"MIN_OPPORTUNITY_SCORE={MIN_OPPORTUNITY_SCORE} (n={len(baseline_trades)})."
    )
    models.insert_recommendation(
        mode=mode,
        metric_name="MIN_OPPORTUNITY_SCORE",
        current_value=MIN_OPPORTUNITY_SCORE,
        recommended_value=best_threshold,
        rationale=rationale,
        sample_size=best_sample_size,
    )
    return [{"metric_name": "MIN_OPPORTUNITY_SCORE", "recommended_value": best_threshold, "rationale": rationale}]
