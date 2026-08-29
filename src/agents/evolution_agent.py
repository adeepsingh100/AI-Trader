"""Hourly: evaluate the active strategy version against
src/learning/promotion_gate.py's multi-dimensional PROMOTE/REJECT/
EXTEND_VALIDATION decision, and auto-promote on PROMOTE — no human-
approval step (Scientific Strategy Optimization Framework, extended;
PROJECT_SPEC.md §2/§3e). Auto-promotion itself isn't new; what changed is
the bar: a candidate now has to clear sample-size floors, risk/
statistical/Monte-Carlo gates, regime/symbol robustness, and a
significant same-market-data improvement over the current real-mode
champion, not just 5 simple thresholds. See promotion_gate.py's module
docstring for the full gate list and PROJECT_SPEC.md §3e for the audit
trail (promotion_audit) and automatic-rollback (strategy_health.py) that
go with it.

Previously also asked an LLM to freely rewrite the strategy's prompt_text/
params_json every night and auto-promoted to real trading the instant 3
simple thresholds cleared — retired entirely. That was unconstrained
"parameter tuning by LLM guesswork," not learning, and the only place in
this codebase real money moved with zero rigorous validation. Strategy
evolution (including stop_loss_pct/take_profit_pct, previously only ever
LLM-guessed) now happens exclusively through the statistically-rigorous
adaptive_strategy_versions candidate pipeline (src/learning/
recommendations.py + simulation.py, orchestrated by
src/learning/adaptive_strategy_engine.py). No LLM call in this module —
trading itself makes none either (src/orchestrator.py); the one place an
LLM is still used is an hourly, code-validated exit-params proposal in
that same recommendations.py pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.agents.risk_manager import today_ist
from src.config import (
    LEARNING_HISTORY_WINDOW_DAYS,
    OPERATIONAL_LOG_RETENTION_DAYS,
    PROMOTION_MAX_DRAWDOWN_PCT,
    PROMOTION_MIN_CUMULATIVE_PNL,
    PROMOTION_MIN_PAPER_DAYS,
    STRATEGY_PROFILES,
)
from src.db import models
from src.resilience import log_fail_open
from src.utils import max_drawdown_pct, parse_timestamp

# Local imports (inside run_evolution, not here) — src.learning.statistics
# imports compute_metrics from this file, so importing it (or anything
# that imports it, e.g. learning_status.py via evidence_engine.py) at
# module level here would be circular.


def compute_metrics(trades: list[dict], capital_to_use: float) -> dict:
    closed = [t for t in trades if t.get("pnl") is not None]
    wins = [t["pnl"] for t in closed if t["pnl"] > 0]
    losses = [t["pnl"] for t in closed if t["pnl"] <= 0]

    return {
        "trades_count": len(closed),
        "win_rate": len(wins) / len(closed) if closed else 0.0,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "cumulative_pnl": sum(t["pnl"] for t in closed),
        "max_drawdown_pct": _max_drawdown_pct(closed, capital_to_use),
    }


def _max_drawdown_pct(closed_trades: list[dict], capital_to_use: float) -> float:
    ordered = sorted(closed_trades, key=lambda t: t["closed_at"])
    return max_drawdown_pct([t["pnl"] for t in ordered], capital_to_use)


def _created_date(version: dict):
    raw = version["created_at"]
    if isinstance(raw, str):
        return parse_timestamp(raw).date()
    return raw.date()


def promotion_ready(version: dict, metrics: dict) -> bool:
    days_live = (today_ist() - _created_date(version)).days
    if days_live < PROMOTION_MIN_PAPER_DAYS:
        return False
    if metrics["cumulative_pnl"] < PROMOTION_MIN_CUMULATIVE_PNL:
        return False
    if metrics["max_drawdown_pct"] > PROMOTION_MAX_DRAWDOWN_PCT:
        return False
    return True


def run_evolution(mode: str = "paper") -> dict:
    """Promotion monitor, looped over every strategy_type with a seeded
    capital_config row for this mode (same "ships dormant" activation gate
    as orchestrator.run_cycle — a type with no capital_config never runs).
    Data Retention is global (not per-strategy_type — it's disk management,
    not a learning concern) and runs once regardless of how many types are
    active. Returns {strategy_type: {...}} — see _run_evolution_for_strategy_type
    for what each entry contains."""
    active_types = [t for t in models.get_active_strategy_types(mode) if t in STRATEGY_PROFILES]
    if not active_types:
        raise RuntimeError(f"no capital_config row for mode={mode!r} — insert one first")

    results = {}
    for strategy_type in active_types:
        results[strategy_type] = _run_evolution_for_strategy_type(mode, strategy_type)

    # Data Retention (runs hourly, piggybacked on this already-scheduled
    # step rather than a new cron job — see src/db/models.py::purge_old_data
    # and src/config.py's Data Retention section). Fails open: a purge
    # error must never block the promotion-monitor logic above it. Global,
    # not per-strategy_type -- these tables aren't strategy-scoped filters,
    # they're disk-management cutoffs across the whole table.
    try:
        now = datetime.now(timezone.utc)
        learning_cutoff = now - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
        operational_cutoff = now - timedelta(days=OPERATIONAL_LOG_RETENTION_DAYS)
        purged = models.purge_old_data(
            {
                "opportunity_evaluations": learning_cutoff,
                "confidence_calibration": learning_cutoff,
                "agent_logs": operational_cutoff,
                "model_usage": operational_cutoff,
                "system_metrics": operational_cutoff,
                "data_quality_log": operational_cutoff,
            }
        )
        if any(purged.values()):
            models.log_agent_event("evolution_agent", "info", f"data retention purge: {purged}")
    except Exception as e:
        log_fail_open("evolution_agent.purge_old_data", e)

    return results


def _run_evolution_for_strategy_type(mode: str, strategy_type: str) -> dict:
    """Promotion monitor for the one live strategy_versions row of this
    strategy_type — no LLM call, no new version creation (that's
    adaptive_strategy_engine.py's candidate pipeline). Delegates the
    actual decision to src/learning/promotion_gate.py::evaluate_promotion —
    PROMOTE auto-flips promoted_to_real, no human click, but only once
    every gate there (sample sizes, risk/statistical/Monte-Carlo, regime/
    symbol robustness, champion improvement) clears; REJECT/EXTEND_VALIDATION
    leave the version exactly as it was. Every evaluation (not just a
    promotion) is written to promotion_audit — see that function's
    docstring for the full gate list. trades are scoped to this specific
    version's own history, so a freshly created version always starts its
    own PROMOTION_MIN_PAPER_DAYS clock at zero regardless."""
    capital_config = models.get_capital_config(mode, strategy_type)
    if capital_config is None:
        raise RuntimeError(f"no capital_config row for mode={mode!r} strategy_type={strategy_type!r}")

    version = models.get_latest_version(strategy_type)
    if version is None:
        raise RuntimeError(f"no strategy_versions row for strategy_type={strategy_type!r} — create one first")

    trades = models.get_closed_trades(mode, version["id"])
    metrics = compute_metrics(trades, capital_config["capital_to_use"])

    # Local imports — src.learning.statistics/fitness/learning_status/
    # promotion_gate ultimately import compute_metrics/promotion_ready
    # from this file, so importing them at module level here would be
    # circular.
    from src.learning.fitness import compute_fitness_score
    from src.learning.learning_status import compute_learning_status
    from src.learning.promotion_gate import build_symbol_to_pair, evaluate_promotion
    from src.learning.statistics import compute_bucket_statistics

    bucket_stats = compute_bucket_statistics(trades, capital_config["capital_to_use"])
    fitness = compute_fitness_score(bucket_stats, capital_config["capital_to_use"])

    eligible = False
    promoted = False
    decision = None
    if mode == "paper" and not version["promoted_to_real"]:
        champion = models.get_latest_promoted_version(strategy_type)
        symbol_to_pair = build_symbol_to_pair(mode, strategy_type)
        decision = evaluate_promotion(
            mode,
            version,
            trades,
            capital_config["capital_to_use"],
            champion=champion,
            symbol_to_pair=symbol_to_pair,
            strategy_type=strategy_type,
        )
        eligible = decision.decision == "PROMOTE"
        if eligible != version.get("promotion_eligible"):
            models.set_strategy_version_promotion_eligible(version["id"], eligible)
        if eligible:
            models.promote_version(version["id"])
            promoted = True
        models.insert_promotion_audit(
            mode=mode,
            event_type="promotion" if eligible else "evaluation",
            decision=decision.decision,
            candidate_version_id=version["id"],
            previous_champion_id=champion["id"] if champion else None,
            new_champion_id=version["id"] if eligible else (champion["id"] if champion else None),
            promotion_score=decision.promotion_score,
            gates=decision.gates,
            breakdown=decision.breakdown,
            reasons=decision.reasons,
            strategy_type=strategy_type,
        )

    learning_status = compute_learning_status(mode, strategy_type)

    models.log_agent_event(
        "evolution_agent",
        "info",
        f"strategy_type={strategy_type} stage={learning_status.stage} "
        f"trades_collected={learning_status.trades_collected} "
        f"evidence_readiness={learning_status.evidence_readiness_pct:.0f}% "
        f"metrics={metrics} fitness_score={fitness['fitness_score']} "
        f"promotion_decision={decision.decision if decision else 'n/a'} promotion_eligible={eligible}"
        + (f" AUTO-PROMOTED version_id={version['id']} to real trading" if promoted else ""),
    )

    return {
        "metrics": metrics,
        "fitness": fitness,
        "promotion_decision": decision.decision if decision else None,
        "promotion_eligible": eligible,
        "promoted": promoted,
        "learning_status": learning_status,
    }


if __name__ == "__main__":
    run_evolution()
