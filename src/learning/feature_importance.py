"""Point-biserial correlation between Feature Engine/Opportunity Scorer
values and trade win/loss outcome. Hand-rolled Pearson-correlation sums,
not stdlib statistics.correlation() — that function doesn't exist before
Python 3.10, and this repo's local dev interpreter is 3.9 even though CI
runs 3.11.

Gated behind RECOMMENDATION_MIN_SAMPLE_SIZE: below that, correlation over
a handful of trades is noise that would read as an authoritative number,
so the write is skipped entirely rather than stored misleadingly.

Two things get correlated here, sharing the same math and the same
feature_importance table (distinguished by the `timeframe` column):
- compute_feature_importance(): raw Feature Engine keys (rsi, adx, ...),
  per configured timeframe (default: primary timeframe only, same as
  before this was generalized).
- compute_subscore_correlation_weights(): the 5 Opportunity Scorer
  sub-scores (trend/momentum/volume/volatility/risk), always tagged with
  the explicit sentinel timeframe "blended" since they're already blended
  across timeframes by score_opportunity(). This one feeds three
  consumers without recomputing three times: the weight-recommendation
  generator (recommendations.py), the nightly cache write here, and
  trade_memory.py's live (but cheap, read-only) similarity weighting."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from statistics import stdev

from src.config import LEARNING_HISTORY_WINDOW_DAYS, RECOMMENDATION_MIN_SAMPLE_SIZE
from src.features.feature_engine import FEATURE_KEYS
from src.features.opportunity_scorer import PRIMARY_TIMEFRAME, weighted_average
from src.db import models
from src.learning.statistics import z_test_two_means

# Non-numeric / not meaningfully correlatable as a plain float.
_EXCLUDED_FEATURE_KEYS = {"volatility_regime", "volume_spike", "obv_rising"}

# The 5 Opportunity Scorer sub-scores, already blended across timeframes —
# same keys score_opportunity() returns, same keys trade_memory.py's
# similarity distance compares.
SUB_SCORE_KEYS = ("trend_score", "momentum_score", "volume_score", "volatility_score", "risk_score")
BLENDED_TIMEFRAME = "blended"


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    sum_x, sum_y = sum(xs), sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)
    sum_y2 = sum(y * y for y in ys)
    denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    return (n * sum_xy - sum_x * sum_y) / denominator if denominator else None


def compute_feature_importance(mode: str, timeframes: list[str] | None = None) -> list[dict]:
    """Nightly/periodic (called from evolution_agent's cron), not per
    10-minute cycle — this is a batch statistical pass over a growing
    dataset, not something the trading loop needs fresh every cycle.

    `timeframes` defaults to [PRIMARY_TIMEFRAME] (this function's original
    behavior). Passing FEATURE_TIMEFRAMES computes per-(timeframe, feature)
    correlation using each trade's already-stored
    entry_eval["features"][tf] raw values — zero new API/candle calls,
    since the full multi-timeframe feature dump is already in the
    opportunity_evaluations.features JSONB from when the trade was scored."""
    timeframes = timeframes if timeframes is not None else [PRIMARY_TIMEFRAME]

    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    trades = [t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None]
    if len(trades) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return []

    outcomes = {t["id"]: (1.0 if t["pnl"] > 0 else 0.0) for t in trades}
    entry_evals = {}
    for trade in trades:
        entry_eval = models.get_entry_evaluation_for_trade(trade["id"])
        if entry_eval is not None:
            entry_evals[trade["id"]] = entry_eval

    results = []
    for timeframe in timeframes:
        per_feature_pairs: dict[str, list[tuple[float, float]]] = {k: [] for k in FEATURE_KEYS}
        for trade_id, entry_eval in entry_evals.items():
            tf_features = (entry_eval.get("features") or {}).get(timeframe) or {}
            won = outcomes[trade_id]
            for key in FEATURE_KEYS:
                if key in _EXCLUDED_FEATURE_KEYS:
                    continue
                value = tf_features.get(key)
                if isinstance(value, (int, float)):
                    per_feature_pairs[key].append((float(value), won))

        for feature_name, pairs in per_feature_pairs.items():
            if len(pairs) < RECOMMENDATION_MIN_SAMPLE_SIZE:
                continue
            xs, ys = zip(*pairs)
            correlation = pearson_correlation(list(xs), list(ys))
            if correlation is None:
                continue
            models.upsert_feature_importance(mode, feature_name, correlation, len(pairs), timeframe)
            results.append(
                {
                    "feature_name": feature_name,
                    "timeframe": timeframe,
                    "correlation_score": correlation,
                    "sample_count": len(pairs),
                }
            )

    return results


def compute_subscore_correlation_weights(
    mode: str, trades: list[dict] | None = None, cache: bool = True
) -> dict[str, float] | None:
    """Correlates the 5 already-flat opportunity_evaluations sub-score
    columns against win/loss, normalizes positive correlations into a
    weight distribution (negatives floored at 0), and — when `cache` is
    True (the default, mode-wide call) — caches the raw correlations in
    feature_importance (timeframe="blended") for trade_memory.py's live
    similarity weighting to read back cheaply.

    `trades` defaults to None, meaning "fetch mode's closed trades over
    LEARNING_HISTORY_WINDOW_DAYS" (the original behavior). Passing an
    explicit (already-filtered) list lets callers scope this to a subset —
    e.g. recommendations.generate_regime_recommendations() scoping to one
    regime's trades — without duplicating the correlation math. Scoped
    calls pass cache=False so a regime-specific correlation never
    overwrites the mode-wide cache trade_memory.py reads.

    Real limitation, stated plainly: this correlation is computed only
    over trades that already cleared the CURRENT OPPORTUNITY_WEIGHT_*
    weights — a form of selection bias this codebase already confesses for
    thresholds in recommendations.py, restated here since it applies to
    weights too. It can only ever ask "of the trades we took, which
    sub-scores separated winners from losers" — never "would a different
    weight set have accepted different trades."

    Returns None if there isn't enough data or every correlation is <= 0
    (nothing to recommend from zero/negative signal)."""
    if trades is None:
        since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
        trades = [t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None]
    if len(trades) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return None

    per_key_pairs: dict[str, list[tuple[float, float]]] = {k: [] for k in SUB_SCORE_KEYS}
    for trade in trades:
        entry_eval = models.get_entry_evaluation_for_trade(trade["id"])
        if entry_eval is None:
            continue
        won = 1.0 if trade["pnl"] > 0 else 0.0
        for key in SUB_SCORE_KEYS:
            value = entry_eval.get(key)
            if isinstance(value, (int, float)):
                per_key_pairs[key].append((float(value), won))

    correlations: dict[str, float] = {}
    for key, pairs in per_key_pairs.items():
        if len(pairs) < RECOMMENDATION_MIN_SAMPLE_SIZE:
            continue
        xs, ys = zip(*pairs)
        correlation = pearson_correlation(list(xs), list(ys))
        if correlation is not None:
            correlations[key] = correlation
            if cache:
                models.upsert_feature_importance(mode, key, correlation, len(pairs), BLENDED_TIMEFRAME)

    if not correlations:
        return None

    positive = {k: max(0.0, v) for k, v in correlations.items()}
    total = sum(positive.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in positive.items()}


def score_separation_p_value(trades: list[dict], weights: dict[str, float]) -> dict | None:
    """For an already-taken trade set, recomputes each trade's opportunity
    score under `weights` (via the same weighted_average() the live
    scorer uses) and tests whether winners and losers separate by score
    significantly better than chance (two-sample z-test on the two
    groups' mean recomputed score). Used to compare a candidate weight
    set against the current one — "does this weighting separate winners
    from losers better" — never to discover trades that weren't taken.

    None if either group has fewer than 2 trades or too few trades have a
    usable entry_eval, matching this module's degrade-to-None convention
    throughout."""
    win_scores, loss_scores = [], []
    for trade in trades:
        if trade.get("pnl") is None:
            continue
        entry_eval = models.get_entry_evaluation_for_trade(trade["id"])
        if entry_eval is None:
            continue
        score = weighted_average({k: entry_eval.get(k) for k in SUB_SCORE_KEYS}, weights)
        if score is None:
            continue
        (win_scores if trade["pnl"] > 0 else loss_scores).append(score)

    if len(win_scores) < 2 or len(loss_scores) < 2:
        return None

    mean_win, mean_loss = sum(win_scores) / len(win_scores), sum(loss_scores) / len(loss_scores)
    p_value = z_test_two_means(
        mean_win, stdev(win_scores), len(win_scores), mean_loss, stdev(loss_scores), len(loss_scores)
    )
    return {
        "p_value": p_value,
        "mean_win_score": mean_win,
        "mean_loss_score": mean_loss,
        "n_win": len(win_scores),
        "n_loss": len(loss_scores),
    }
