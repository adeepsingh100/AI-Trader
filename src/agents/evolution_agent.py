"""Nightly: score the active strategy version's paper trades and flag it
promotion_eligible if it clears the promotion bar — a human still reviews
and flips promoted_to_real themselves (Scientific Strategy Optimization
Framework, PROJECT_SPEC.md §2/§3).

Previously also asked an LLM to freely rewrite the strategy's prompt_text/
params_json every night and auto-promoted to real trading the instant 3
simple thresholds cleared — retired entirely. That was unconstrained
"parameter tuning by LLM guesswork," not learning, and the only place in
this codebase real money moved with zero human approval. Strategy
evolution (including stop_loss_pct/take_profit_pct, previously only ever
LLM-guessed) now happens exclusively through the statistically-rigorous,
human-approved adaptive_strategy_versions candidate pipeline
(src/learning/recommendations.py + simulation.py, orchestrated by
src/learning/adaptive_strategy_engine.py). signal_agent's validation
prompt is stable/human-edited only."""

from __future__ import annotations

from src.agents.risk_manager import today_ist
from src.config import (
    PROMOTION_MAX_DRAWDOWN_PCT,
    PROMOTION_MIN_CUMULATIVE_PNL,
    PROMOTION_MIN_FITNESS_SCORE,
    PROMOTION_MIN_PAPER_DAYS,
)
from src.db import models
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


def promotion_eligible(version: dict, metrics: dict, trades: list[dict], fitness_score: float | None) -> bool:
    """Extends promotion_ready with statistical significance and a fitness
    floor (Scientific Strategy Optimization Framework) — paper
    profitability clearing 3 raw thresholds isn't enough on its own,
    it must also not plausibly be noise (bootstrap CI on trade pnls,
    lower bound still positive) and must clear a minimum blended fitness
    score. Code only sets strategy_versions.promotion_eligible from this —
    a human still reviews and flips promoted_to_real themselves."""
    if not promotion_ready(version, metrics):
        return False
    from src.backtest.statistical_validation import bootstrap_confidence_interval

    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
    ci = bootstrap_confidence_interval(pnls)
    if ci is None or ci["ci_low"] <= 0:
        return False
    if fitness_score is None or fitness_score < PROMOTION_MIN_FITNESS_SCORE:
        return False
    return True


def run_evolution(mode: str = "paper") -> dict:
    """Promotion monitor for the one live strategy_versions row — no LLM
    call, no new version creation (that's adaptive_strategy_engine.py's
    candidate pipeline). Newly clearing promotion_eligible()'s five gates
    auto-flips promoted_to_real — no human click. Safe to automate
    because the gates themselves (paper-days, cumulative PnL, drawdown,
    bootstrap CI, fitness floor) are unchanged; trades are scoped to this
    specific version's own history, so a freshly created version always
    starts its own PROMOTION_MIN_PAPER_DAYS clock at zero regardless."""
    capital_config = models.get_capital_config(mode)
    if capital_config is None:
        raise RuntimeError(f"no capital_config row for mode={mode!r} — insert one first")

    version = models.get_latest_version()
    if version is None:
        raise RuntimeError("no strategy_versions row — create one first")

    trades = models.get_closed_trades(mode, version["id"])
    metrics = compute_metrics(trades, capital_config["capital_to_use"])

    # Local imports — src.learning.statistics/fitness/learning_status
    # ultimately import compute_metrics from this file, so importing them
    # at module level here would be circular.
    from src.learning.fitness import compute_fitness_score
    from src.learning.learning_status import compute_learning_status
    from src.learning.statistics import compute_bucket_statistics

    bucket_stats = compute_bucket_statistics(trades, capital_config["capital_to_use"])
    fitness = compute_fitness_score(bucket_stats, capital_config["capital_to_use"])

    eligible = mode == "paper" and not version["promoted_to_real"] and promotion_eligible(
        version, metrics, trades, fitness["fitness_score"]
    )
    promoted = False
    if eligible != version.get("promotion_eligible"):
        models.set_strategy_version_promotion_eligible(version["id"], eligible)
        if eligible:
            models.promote_version(version["id"])
            promoted = True

    learning_status = compute_learning_status(mode)

    models.log_agent_event(
        "evolution_agent",
        "info",
        f"stage={learning_status.stage} trades_collected={learning_status.trades_collected} "
        f"evidence_readiness={learning_status.evidence_readiness_pct:.0f}% "
        f"metrics={metrics} fitness_score={fitness['fitness_score']} promotion_eligible={eligible}"
        + (f" AUTO-PROMOTED version_id={version['id']} to real trading" if promoted else ""),
    )

    return {
        "metrics": metrics,
        "fitness": fitness,
        "promotion_eligible": eligible,
        "promoted": promoted,
        "learning_status": learning_status,
    }


if __name__ == "__main__":
    run_evolution()
