"""Point-biserial correlation between each raw Feature Engine value
(primary timeframe only — a first pass, not all 4 timeframes x all
features) and trade win/loss outcome. Hand-rolled Pearson-correlation
sums, not stdlib statistics.correlation() — that function doesn't exist
before Python 3.10, and this repo's local dev interpreter is 3.9 even
though CI runs 3.11.

Gated behind RECOMMENDATION_MIN_SAMPLE_SIZE: below that, correlation over
a handful of trades is noise that would read as an authoritative number,
so the write is skipped entirely rather than stored misleadingly."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from src.config import LEARNING_HISTORY_WINDOW_DAYS, RECOMMENDATION_MIN_SAMPLE_SIZE
from src.features.feature_engine import FEATURE_KEYS
from src.features.opportunity_scorer import PRIMARY_TIMEFRAME
from src.db import models

# Non-numeric / not meaningfully correlatable as a plain float.
_EXCLUDED_FEATURE_KEYS = {"volatility_regime", "volume_spike", "obv_rising"}


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    sum_x, sum_y = sum(xs), sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    sum_y2 = sum(y * y for y in ys)
    denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    return (n * sum_xy - sum_x * sum_y) / denominator if denominator else None


def compute_feature_importance(mode: str) -> list[dict]:
    """Nightly/periodic (called from evolution_agent's cron), not per
    10-minute cycle — this is a batch statistical pass over a growing
    dataset, not something the trading loop needs fresh every cycle."""
    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    trades = [t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None]
    if len(trades) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return []

    outcomes = {t["id"]: (1.0 if t["pnl"] > 0 else 0.0) for t in trades}
    per_feature_pairs: dict[str, list[tuple[float, float]]] = {k: [] for k in FEATURE_KEYS}

    for trade in trades:
        entry_eval = models.get_entry_evaluation_for_trade(trade["id"])
        if entry_eval is None:
            continue
        primary_features = (entry_eval.get("features") or {}).get(PRIMARY_TIMEFRAME) or {}
        won = outcomes[trade["id"]]
        for key in FEATURE_KEYS:
            if key in _EXCLUDED_FEATURE_KEYS:
                continue
            value = primary_features.get(key)
            if isinstance(value, (int, float)):
                per_feature_pairs[key].append((float(value), won))

    results = []
    for feature_name, pairs in per_feature_pairs.items():
        if len(pairs) < RECOMMENDATION_MIN_SAMPLE_SIZE:
            continue
        xs, ys = zip(*pairs)
        correlation = _pearson_correlation(list(xs), list(ys))
        if correlation is None:
            continue
        models.upsert_feature_importance(mode, feature_name, correlation, len(pairs))
        results.append({"feature_name": feature_name, "correlation_score": correlation, "sample_count": len(pairs)})

    return results
