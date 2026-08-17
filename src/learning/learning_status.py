"""Progressive Learning Stages. One function, one source of truth for "how
much evidence has this mode collected and what is the learning engine doing
about it" — consumed by evolution_agent.py's and adaptive_strategy_engine.py's
reports, reports.py's HTML, and mirrored by the dashboard.

Deliberately depends on nothing that imports evolution_agent.py (statistics.py
and fitness.py both do, for compute_metrics) — only models + config — so this
is safely importable at module level from both engines with no circular-import
workaround needed.

Stage boundaries mirror the 4 LEARNING_STAGE_*_MIN_TRADES config constants,
which are themselves the actual outer gates recommendations.py/simulation.py/
weakness_detection.py check — this module doesn't gate anything itself, it
only reports where a mode currently sits relative to those same gates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import (
    LEARNING_HISTORY_WINDOW_DAYS,
    LEARNING_STAGE_HYPOTHESIS_MIN_TRADES,
    LEARNING_STAGE_OBSERVATION_MIN_TRADES,
    LEARNING_STAGE_SIMULATION_MIN_TRADES,
    LEARNING_STAGE_VALIDATION_MIN_TRADES,
)
from src.db import models

_ACTIVITY = {
    "BOOTSTRAP": "Collecting trade data, rejection reasons, and feature distributions only. No analysis yet.",
    "OBSERVATION": "Analyzing rejection reasons, feature distributions, and weakness patterns. No strategy changes yet.",
    "HYPOTHESIS": "Generating hypotheses (weight/threshold/exit-parameter recommendations) from observed weaknesses. No candidate strategies yet.",
    "SIMULATION": "Testing hypotheses via backtest and walk-forward simulation. Candidates are validated but not yet created.",
    "VALIDATION": "Full validation active — passing simulations create candidate strategies, pending human approval for promotion.",
}


def _stage_for(trades_collected: int) -> tuple[str, str | None, int | None]:
    """(stage, next_stage, next_stage_min_trades) — next_stage is None once
    trades_collected clears the final boundary (VALIDATION, 500+)."""
    if trades_collected < LEARNING_STAGE_OBSERVATION_MIN_TRADES:
        return "BOOTSTRAP", "OBSERVATION", LEARNING_STAGE_OBSERVATION_MIN_TRADES
    if trades_collected < LEARNING_STAGE_HYPOTHESIS_MIN_TRADES:
        return "OBSERVATION", "HYPOTHESIS", LEARNING_STAGE_HYPOTHESIS_MIN_TRADES
    if trades_collected < LEARNING_STAGE_SIMULATION_MIN_TRADES:
        return "HYPOTHESIS", "SIMULATION", LEARNING_STAGE_SIMULATION_MIN_TRADES
    if trades_collected < LEARNING_STAGE_VALIDATION_MIN_TRADES:
        return "SIMULATION", "VALIDATION", LEARNING_STAGE_VALIDATION_MIN_TRADES
    return "VALIDATION", None, None


def _reason_for(stage: str, trades_collected: int, next_stage: str | None, next_min: int | None) -> str:
    if next_stage is None:
        return f"Full validation stage reached ({trades_collected} trades collected)."
    remaining = next_min - trades_collected
    return f"{remaining} more closed trade(s) needed to reach {next_stage} (requires {next_min})."


def compute_learning_status(mode: str) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    closed = [t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None]
    trades_collected = len(closed)
    winning_trades = sum(1 for t in closed if t["pnl"] > 0)
    losing_trades = trades_collected - winning_trades
    rejected_trades = len(models.get_hold_evaluations_since(mode, since))

    stage, next_stage, next_min = _stage_for(trades_collected)
    trades_to_next_stage = max(0, next_min - trades_collected) if next_min is not None else 0

    version = models.get_latest_version()
    promotion_eligible = bool(version and version.get("promotion_eligible"))

    return {
        "stage": stage,
        "trades_collected": trades_collected,
        "rejected_trades": rejected_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "data_sufficiency_pct": min(100.0, trades_collected / LEARNING_STAGE_VALIDATION_MIN_TRADES * 100),
        "recommendations_count": len(models.get_recommendations(mode)),
        "simulations_count": len(models.get_strategy_simulations(mode)),
        "candidates_count": len(models.get_adaptive_strategy_versions(mode)),
        "promotion_eligible": promotion_eligible,
        "next_stage": next_stage,
        "trades_to_next_stage": trades_to_next_stage,
        "current_activity": _ACTIVITY[stage],
        "reason": _reason_for(stage, trades_collected, next_stage, next_min),
    }
