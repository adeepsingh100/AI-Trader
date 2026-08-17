"""Weakness Detection (Step 3, Scientific Strategy Optimization
Framework). A thin reporting layer over tables the Learning Engine already
populates nightly — learning_statistics already has per-dimension stats
(symbol/market_regime/opportunity_score_bucket/confidence_bucket/
strategy_version/weekday/hour/exit_reason), feature_importance already has
point-biserial correlation-ranked indicators. Nothing here recomputes
statistics; it only ranks what's already stored."""

from __future__ import annotations

from src.config import LEARNING_STAGE_OBSERVATION_MIN_TRADES
from src.db import models


def _summarize(row: dict) -> dict:
    return {
        "value": row["dimension_value"],
        "expectancy": row["expectancy"],
        "trades_count": row["trades_count"],
    }


def identify_weaknesses(mode: str) -> dict:
    """worst/best bucket per dimension_type (gated on
    LEARNING_STAGE_OBSERVATION_MIN_TRADES — Progressive Learning Stages,
    Stage 1 OBSERVATION's flagship output, ranked by expectancy) plus
    worst/best indicator by correlation magnitude — never fabricated from a
    thin bucket, same "None below the sample floor" policy as everywhere
    else in src/learning/."""
    stats_rows = models.get_learning_statistics(mode)
    dimension_types = {r["dimension_type"] for r in stats_rows}

    worst_by_dimension, best_by_dimension = {}, {}
    for dimension_type in dimension_types:
        eligible = [
            r
            for r in stats_rows
            if r["dimension_type"] == dimension_type
            and (r.get("trades_count") or 0) >= LEARNING_STAGE_OBSERVATION_MIN_TRADES
            and r.get("expectancy") is not None
        ]
        if not eligible:
            continue
        worst_by_dimension[dimension_type] = _summarize(min(eligible, key=lambda r: r["expectancy"]))
        best_by_dimension[dimension_type] = _summarize(max(eligible, key=lambda r: r["expectancy"]))

    # Raw Feature Engine indicators only (excludes the "blended" sentinel
    # timeframe, which correlates the 5 opportunity-scorer sub-scores, a
    # related but different ranking).
    indicators = [
        r
        for r in models.get_feature_importance(mode)
        if r.get("timeframe") != "blended" and (r.get("sample_count") or 0) >= LEARNING_STAGE_OBSERVATION_MIN_TRADES
    ]
    worst_indicator = min(indicators, key=lambda r: r["correlation_score"], default=None)
    best_indicator = max(indicators, key=lambda r: r["correlation_score"], default=None)

    return {
        "worst_by_dimension": worst_by_dimension,
        "best_by_dimension": best_by_dimension,
        "worst_indicator": worst_indicator,
        "best_indicator": best_indicator,
    }
