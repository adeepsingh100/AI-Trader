"""Feature Drift Detection (Step 6, PROJECT_SPEC.md §3d). Compares a
recent window against an older baseline window for feature-value
distribution, feature-importance trend, and prediction accuracy/confidence-
calibration/win-rate/opportunity-score-accuracy trend — advisory only,
writes to drift_alerts, never touches config.py or scoring weights.

Runs as its own independent step in evolution.yml (a third `run:` line,
after adaptive_strategy_engine — never merged into
evolution_agent.run_evolution() or adaptive_strategy_engine.py itself, per
this codebase's own "don't couple independent learning steps" rule, which
those two already follow).

Population Stability Index is hand-rolled (bucketed frequency ratio), not
scipy — same "no numpy/scipy" discipline as the rest of src/learning/."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from src.config import (
    DRIFT_BASELINE_WINDOW_DAYS,
    DRIFT_PSI_BUCKET_COUNT,
    DRIFT_PSI_CRITICAL_THRESHOLD,
    DRIFT_PSI_WARNING_THRESHOLD,
    DRIFT_RECENT_WINDOW_DAYS,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
    SIGNIFICANCE_THRESHOLD,
)
from src.db import models
from src.features.feature_engine import FEATURE_KEYS
from src.features.opportunity_scorer import PRIMARY_TIMEFRAME
from src.learning.feature_importance import _EXCLUDED_FEATURE_KEYS
from src.learning.statistics import z_test_two_proportions

_NUMERIC_FEATURE_KEYS = [k for k in FEATURE_KEYS if k not in _EXCLUDED_FEATURE_KEYS]


def _severity(magnitude: float) -> str | None:
    if magnitude >= DRIFT_PSI_CRITICAL_THRESHOLD:
        return "critical"
    if magnitude >= DRIFT_PSI_WARNING_THRESHOLD:
        return "warning"
    return None


def population_stability_index(baseline: list[float], recent: list[float], buckets: int = DRIFT_PSI_BUCKET_COUNT) -> float | None:
    if len(baseline) < 2 or len(recent) < 2:
        return None
    combined = baseline + recent
    lo, hi = min(combined), max(combined)
    if hi == lo:
        return 0.0
    width = (hi - lo) / buckets

    def fractions(values: list[float]) -> list[float]:
        counts = [0] * buckets
        for v in values:
            idx = min(buckets - 1, max(0, int((v - lo) / width)))
            counts[idx] += 1
        return [c / len(values) for c in counts]

    eps = 1e-6
    psi = 0.0
    for bf, rf in zip(fractions(baseline), fractions(recent)):
        bf, rf = bf or eps, rf or eps
        psi += (rf - bf) * math.log(rf / bf)
    return psi


def _split_by_window(trades: list[dict], now: datetime) -> tuple[list[dict], list[dict]]:
    cutoff = now - timedelta(days=DRIFT_RECENT_WINDOW_DAYS)
    recent, baseline = [], []
    for t in trades:
        closed_at = datetime.fromisoformat(t["closed_at"].replace("Z", "+00:00"))
        (recent if closed_at >= cutoff else baseline).append(t)
    return baseline, recent


def detect_feature_drift(mode: str, timeframe: str = PRIMARY_TIMEFRAME) -> list[dict]:
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=DRIFT_BASELINE_WINDOW_DAYS)
    trades = [t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None]
    baseline_trades, recent_trades = _split_by_window(trades, now)

    def values_for(trade_list: list[dict], key: str) -> list[float]:
        out = []
        for t in trade_list:
            entry_eval = models.get_entry_evaluation_for_trade(t["id"])
            if not entry_eval:
                continue
            v = (entry_eval.get("features") or {}).get(timeframe, {}).get(key)
            if v is not None:
                out.append(float(v))
        return out

    alerts = []
    for key in _NUMERIC_FEATURE_KEYS:
        baseline_values = values_for(baseline_trades, key)
        recent_values = values_for(recent_trades, key)
        psi = population_stability_index(baseline_values, recent_values)
        if psi is None:
            continue
        severity = _severity(psi)
        if severity is None:
            continue
        alerts.append(
            models.insert_drift_alert(
                component="feature_engine",
                drift_type=f"feature_distribution:{key}",
                severity=severity,
                baseline_value=None,
                recent_value=psi,
                detail={"timeframe": timeframe, "psi": psi, "baseline_n": len(baseline_values), "recent_n": len(recent_values)},
            )
        )
    return alerts


def detect_feature_importance_drift(mode: str, timeframe: str = PRIMARY_TIMEFRAME) -> list[dict]:
    """feature_importance rows are nightly snapshots (already timestamped);
    compares the two most-recent rows per feature_key rather than raw
    values (a PSI over point correlations wouldn't mean anything) — a
    correlation-magnitude delta beyond the PSI thresholds (reused, not a
    third redundant constant pair — the two metrics live on a comparable
    0-1ish scale) counts as drift."""
    rows = models.get_feature_importance(mode, timeframe=timeframe)
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r["feature_key"], []).append(r)

    alerts = []
    for key, key_rows in by_key.items():
        ordered = sorted(key_rows, key=lambda r: r["computed_at"], reverse=True)
        if len(ordered) < 2:
            continue
        recent_corr, baseline_corr = ordered[0].get("correlation"), ordered[-1].get("correlation")
        if recent_corr is None or baseline_corr is None:
            continue
        delta = abs(recent_corr - baseline_corr)
        severity = _severity(delta)
        if severity is None:
            continue
        alerts.append(
            models.insert_drift_alert(
                component="feature_importance",
                drift_type=f"feature_importance:{key}",
                severity=severity,
                baseline_value=baseline_corr,
                recent_value=recent_corr,
                detail={"timeframe": timeframe, "delta": delta},
            )
        )
    return alerts


def _proportion_drift_alert(
    component: str, drift_type: str, baseline_trades: list[dict], recent_trades: list[dict], is_win
) -> dict | None:
    if len(baseline_trades) < RECOMMENDATION_MIN_SAMPLE_SIZE or len(recent_trades) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return None
    baseline_wins = sum(1 for t in baseline_trades if is_win(t))
    recent_wins = sum(1 for t in recent_trades if is_win(t))
    baseline_rate = baseline_wins / len(baseline_trades)
    recent_rate = recent_wins / len(recent_trades)
    p_value = z_test_two_proportions(recent_wins, len(recent_trades), baseline_wins, len(baseline_trades))
    if p_value is None or p_value >= SIGNIFICANCE_THRESHOLD or recent_rate >= baseline_rate:
        return None
    drop = baseline_rate - recent_rate
    severity = "critical" if drop >= 0.2 else "warning"
    return models.insert_drift_alert(
        component=component,
        drift_type=drift_type,
        severity=severity,
        baseline_value=baseline_rate,
        recent_value=recent_rate,
        detail={"p_value": p_value, "baseline_n": len(baseline_trades), "recent_n": len(recent_trades)},
    )


def detect_performance_drift(mode: str) -> list[dict]:
    """win_rate + confidence-calibration accuracy + opportunity-score
    accuracy drift, each a proportion compared baseline-vs-recent via the
    existing z_test_two_proportions — reused directly, not reimplemented.
    Only flags a STATISTICALLY SIGNIFICANT (SIGNIFICANCE_THRESHOLD)
    worsening, never an improvement or noise.

    confidence_was_accurate/opportunity_score_was_accurate come from
    trade_evaluations (written by learning/statistics.py's
    process_closed_trades() after every trade closes) — trades without a
    row there yet (not self-evaluated) simply don't contribute to those
    two proportions, same "None over fabrication" rule as everywhere else
    in src/learning/."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=DRIFT_BASELINE_WINDOW_DAYS)
    trades = [t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None]
    baseline_trades, recent_trades = _split_by_window(trades, now)

    evaluations = {
        e["trade_id"]: e
        for e in models.get_trade_evaluations([t["id"] for t in baseline_trades + recent_trades])
    }

    def with_evaluation(trade_list: list[dict], field: str) -> list[dict]:
        return [t for t in trade_list if evaluations.get(t["id"], {}).get(field) is not None]

    alerts = []
    win_rate_alert = _proportion_drift_alert(
        "trading", "win_rate", baseline_trades, recent_trades, lambda t: t["pnl"] > 0
    )
    if win_rate_alert:
        alerts.append(win_rate_alert)

    confidence_alert = _proportion_drift_alert(
        "confidence_calibration",
        "confidence_accuracy",
        with_evaluation(baseline_trades, "confidence_was_accurate"),
        with_evaluation(recent_trades, "confidence_was_accurate"),
        lambda t: evaluations[t["id"]]["confidence_was_accurate"],
    )
    if confidence_alert:
        alerts.append(confidence_alert)

    opportunity_score_alert = _proportion_drift_alert(
        "opportunity_scorer",
        "opportunity_score_accuracy",
        with_evaluation(baseline_trades, "opportunity_score_was_accurate"),
        with_evaluation(recent_trades, "opportunity_score_was_accurate"),
        lambda t: evaluations[t["id"]]["opportunity_score_was_accurate"],
    )
    if opportunity_score_alert:
        alerts.append(opportunity_score_alert)

    return alerts


def run_drift_detection(mode: str = "paper") -> dict:
    feature_alerts = detect_feature_drift(mode)
    importance_alerts = detect_feature_importance_drift(mode)
    performance_alerts = detect_performance_drift(mode)
    return {
        "feature_drift_alerts": len(feature_alerts),
        "feature_importance_drift_alerts": len(importance_alerts),
        "performance_drift_alerts": len(performance_alerts),
    }


if __name__ == "__main__":
    print(run_drift_detection())
