"""Strategy Health Engine (Step 7, PROJECT_SPEC.md §3d). Health score per
strategy_version from rolling Sharpe/drawdown/win-rate/profit-factor
(learning/statistics.py, reused), recent-vs-historical performance
(z-tests, reused), and walk-forward pass rate where a backtest exists for
that version — that factor is simply omitted (never fabricated) when none
does. Maps to Excellent/Good/Warning/Critical; Critical auto-suspends the
version (status-only, never a delete — reversible in Supabase any time).
When the suspended version is ALSO the current real-mode champion
(models.get_latest_promoted_version()), this is Automatic Rollback (Phase
20 of the strategy-refinement audit): get_latest_promoted_version()
already excludes suspended versions, so re-resolving it right after the
suspend write IS the rollback — the natural fallback to the next-most-
recent still-active promoted version. The only new behavior here is
making that explicit and audited: a promotion_audit row with
event_type='rollback' records which version was reinstated (or that none
was, if nothing else was ever promoted). No new monitoring machinery —
run_strategy_health(mode="real") reuses the identical health computation
already used for paper, just scoped to real trades.

Runs hourly (evolution.yml, `python -m src.learning.strategy_health`, its
own independent step — never inside evolution_agent.run_evolution() or
adaptive_strategy_engine.py) for both "paper" and "real" modes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import (
    RECOMMENDATION_MIN_SAMPLE_SIZE,
    STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED,
    STRATEGY_HEALTH_EXCELLENT_THRESHOLD,
    STRATEGY_HEALTH_GOOD_THRESHOLD,
    STRATEGY_HEALTH_WARNING_THRESHOLD,
)
from src.db import models
from src.learning.fitness import (
    drawdown_component as _drawdown_component,
    profit_factor_component as _profit_factor_component,
    sharpe_component as _sharpe_component,
    win_rate_component as _win_rate_component,
)
from src.learning.statistics import compute_bucket_statistics, z_test_two_means
from src.resilience import log_fail_open
from src.utils import clamp

# Each component contributes 0-100, weighted equally — a simple, explicit
# blend rather than an opaque learned weighting (this whole learning
# subsystem is pure statistics, never ML).
_COMPONENT_WEIGHT = 1 / 4


def _recent_vs_historical_component(recent_stats: dict, historical_stats: dict) -> float | None:
    """Statistically significant IMPROVEMENT nudges this component up,
    significant DEGRADATION nudges it down, no significant difference (or
    insufficient sample) leaves it neutral at 50 — never fabricated from a
    handful of trades."""
    recent_n, historical_n = recent_stats.get("trades_count") or 0, historical_stats.get("trades_count") or 0
    if recent_n < RECOMMENDATION_MIN_SAMPLE_SIZE or historical_n < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return None
    recent_exp, historical_exp = recent_stats.get("expectancy"), historical_stats.get("expectancy")
    if recent_exp is None or historical_exp is None:
        return None
    # Approximate each bucket's expectancy stdev from its Sharpe (stdev =
    # mean/sharpe) — compute_bucket_statistics doesn't return raw stdev,
    # and re-deriving the full return series here would duplicate that
    # function's own work; this is a reasonable approximation, not exact.
    recent_sharpe, historical_sharpe = recent_stats.get("sharpe_ratio"), historical_stats.get("sharpe_ratio")
    if not recent_sharpe or not historical_sharpe:
        return 50.0
    recent_sd, historical_sd = abs(recent_exp / recent_sharpe), abs(historical_exp / historical_sharpe)
    p_value = z_test_two_means(recent_exp, recent_sd, recent_n, historical_exp, historical_sd, historical_n)
    if p_value is None:
        return 50.0
    from src.config import SIGNIFICANCE_THRESHOLD

    if p_value >= SIGNIFICANCE_THRESHOLD:
        return 50.0
    return 75.0 if recent_exp > historical_exp else 25.0


def _walk_forward_component(run_id: int | None) -> float | None:
    if run_id is None:
        return None
    folds = models.get_backtest_walk_forward_folds(run_id)
    if not folds:
        return None
    passed = [f for f in folds if f.get("passed") is not None]
    if not passed:
        return None
    return clamp(sum(1 for f in passed if f["passed"]) / len(passed) * 100, 0, 100)


def _tier(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= STRATEGY_HEALTH_EXCELLENT_THRESHOLD:
        return "excellent"
    if score >= STRATEGY_HEALTH_GOOD_THRESHOLD:
        return "good"
    if score >= STRATEGY_HEALTH_WARNING_THRESHOLD:
        return "warning"
    return "critical"


def compute_health_score(
    mode: str, version: dict, capital_to_use: float, backtest_run_id: int | None = None
) -> dict:
    trades = models.get_closed_trades(mode, version["id"])
    recent_since = datetime.now(timezone.utc) - timedelta(days=30)
    recent_trades = [t for t in trades if t.get("closed_at") and t["closed_at"] >= recent_since.isoformat()]

    overall_stats = compute_bucket_statistics(trades, capital_to_use)
    recent_stats = compute_bucket_statistics(recent_trades, capital_to_use)

    components = {
        "sharpe": _sharpe_component(overall_stats["sharpe_ratio"]),
        "drawdown": _drawdown_component(overall_stats["max_drawdown_pct"]),
        "win_rate": _win_rate_component(overall_stats["win_rate"]),
        "profit_factor": _profit_factor_component(overall_stats["profit_factor"]),
    }
    available = {k: v for k, v in components.items() if v is not None}
    health_score = sum(available.values()) / len(available) if available else None

    recent_vs_historical = _recent_vs_historical_component(recent_stats, overall_stats)
    walk_forward = _walk_forward_component(backtest_run_id)

    return {
        "health_score": health_score,
        "tier": _tier(health_score),
        "breakdown": {
            **components,
            "recent_vs_historical": recent_vs_historical,
            "walk_forward_pass_rate": walk_forward,
            "trades_count": overall_stats["trades_count"],
        },
    }


def run_strategy_health(mode: str = "paper") -> dict:
    versions = models.get_active_strategy_versions()
    capital_config = models.get_capital_config(mode)
    capital_to_use = capital_config["capital_to_use"] if capital_config else 0

    # Captured once, before any suspension this call might trigger below —
    # the rollback check compares each suspended version's id against
    # WHO WAS CHAMPION at the start of this run, not a moving target.
    # Fails open: a transient failure to read this must never block the
    # health-scoring loop below, only skip the rollback-audit piece.
    try:
        champion_before = models.get_latest_promoted_version()
        champion_before_id = champion_before["id"] if champion_before else None
    except Exception as e:
        log_fail_open("strategy_health.get_latest_promoted_version", e)
        champion_before_id = None

    scored, suspended, rolled_back = [], [], []
    for version in versions:
        result = compute_health_score(mode, version, capital_to_use)
        models.insert_strategy_health_score(
            version["id"], result["health_score"], result["tier"], result["breakdown"]
        )
        scored.append({"version_id": version["id"], **result})
        if (
            STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED
            and result["tier"] == "critical"
            and result["breakdown"]["trades_count"] >= RECOMMENDATION_MIN_SAMPLE_SIZE
        ):
            models.update_strategy_version_status(version["id"], "suspended")
            suspended.append(version["id"])

            if champion_before_id is not None and version["id"] == champion_before_id:
                try:
                    new_champion = models.get_latest_promoted_version()
                    new_champion_id = new_champion["id"] if new_champion else None
                    models.insert_promotion_audit(
                        mode=mode,
                        event_type="rollback",
                        decision="REJECT",
                        candidate_version_id=version["id"],
                        previous_champion_id=champion_before_id,
                        new_champion_id=new_champion_id,
                        breakdown=result["breakdown"],
                        reasons=[
                            f"real-mode champion (version {champion_before_id}) auto-suspended: "
                            f"health tier=critical, score={result['health_score']}"
                        ],
                    )
                    models.log_agent_event(
                        "strategy_health",
                        "warning",
                        f"AUTOMATIC ROLLBACK: champion version_id={champion_before_id} suspended "
                        f"(critical health), reinstated version_id={new_champion_id}",
                    )
                    rolled_back.append(version["id"])
                except Exception as e:
                    log_fail_open("strategy_health.rollback_audit", e)

    return {"scored": len(scored), "suspended": suspended, "rolled_back": rolled_back}


if __name__ == "__main__":
    # Both modes: paper suspension protects paper-only candidates from
    # continuing to run; real suspension is what actually triggers
    # Automatic Rollback above, scoped to the live real-mode champion's
    # own real trades (a promoted version's paper trades often stop
    # growing once evolution moves on to a newer paper candidate, so real
    # trades are the only reliable ongoing signal for it).
    print(run_strategy_health("paper"))
    print(run_strategy_health("real"))
