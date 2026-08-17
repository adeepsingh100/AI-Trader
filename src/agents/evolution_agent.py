"""Nightly: score the active strategy version's paper trades, ask the LLM
for an improved prompt/params (saved as a new immutable version), and
promote the version just scored to real trading if it clears the
configurable promotion bar. See PROJECT_SPEC.md §2 and §3."""

from __future__ import annotations

import json

from src.agents.risk_manager import today_ist
from src.config import (
    PROMOTION_MAX_DRAWDOWN_PCT,
    PROMOTION_MIN_CUMULATIVE_PNL,
    PROMOTION_MIN_PAPER_DAYS,
)
from src.db import models
from src.groq_client import AllModelsFailedError, chat
from src.lenient_json import parse_llm_json
from src.utils import max_drawdown_pct, parse_timestamp

# Local imports (inside run_evolution, not here) — both learning-engine
# modules import compute_metrics from this file, so importing them at
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


def propose_next_version(metrics: dict, current_prompt: str, current_params: dict):
    messages = [
        {
            "role": "system",
            "content": (
                "You are tuning a crypto trading strategy prompt based on its "
                "recent paper-trading performance. Propose an improved prompt "
                "and params. params_json.stop_loss_pct and take_profit_pct, if "
                "set, are enforced automatically as decimal fractions of entry "
                "price (0.02 means 2%) — a position is closed the moment the "
                "price moves against or in favor of entry by that fraction, "
                "independent of your own buy/sell signals. Omit either key to "
                "leave that side unenforced. Respond as JSON only: "
                '{"prompt_text": "...", "params_json": {...}, "notes": "..."}'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "current_prompt": current_prompt,
                    "current_params": current_params,
                    "metrics": metrics,
                }
            ),
        },
    ]
    # A rewritten strategy prompt + params + notes runs longer than a
    # signal's one-line reasoning — more headroom than chat()'s default,
    # still well inside the 8K total budget alongside this call's small input.
    try:
        content, events = chat(messages, max_tokens=2048)
    except AllModelsFailedError as e:
        return {
            "prompt_text": current_prompt,
            "params_json": current_params,
            "notes": f"LLM call failed, carried forward unchanged: {e}",
        }, e.events

    try:
        proposal = parse_llm_json(content)
    except (json.JSONDecodeError, TypeError):
        proposal = {
            "prompt_text": current_prompt,
            "params_json": current_params,
            "notes": f"LLM proposal unparseable, carried forward unchanged: {content!r}",
        }
    return proposal, events


def run_evolution(mode: str = "paper") -> dict:
    capital_config = models.get_capital_config(mode)
    if capital_config is None:
        raise RuntimeError(f"no capital_config row for mode={mode!r} — insert one first")

    version = models.get_latest_version()
    if version is None:
        raise RuntimeError("no strategy_versions row — create one first")

    trades = models.get_closed_trades(mode, version["id"])
    metrics = compute_metrics(trades, capital_config["capital_to_use"])

    proposal, usage_events = propose_next_version(
        metrics, version["prompt_text"], version.get("params_json") or {}
    )
    models.log_model_usage(usage_events)

    new_version = models.insert_strategy_version(
        version_number=version["version_number"] + 1,
        prompt_text=proposal["prompt_text"],
        params_json=proposal.get("params_json") or {},
        notes=proposal.get("notes"),
    )

    promoted = False
    if mode == "paper" and not version["promoted_to_real"] and promotion_ready(version, metrics):
        models.promote_version(version["id"])
        promoted = True

    # Batch/periodic learning-engine passes — piggyback on this nightly
    # cron rather than adding a new workflow. Both are no-ops (empty list)
    # below their own sample-size floors, so this is cheap and safe on a
    # young dataset. Imported here, not at module level — both modules
    # import compute_metrics from this file, so a module-level import
    # here would be circular.
    from src.learning.feature_importance import compute_feature_importance
    from src.learning.recommendations import generate_recommendations

    feature_importance = compute_feature_importance(mode)
    recommendations = generate_recommendations(mode)

    models.log_agent_event(
        "evolution_agent",
        "info",
        f"metrics={metrics} promoted={promoted} new_version={new_version['version_number']} "
        f"feature_importance_rows={len(feature_importance)} recommendations={len(recommendations)}",
    )

    return {
        "metrics": metrics,
        "new_version": new_version,
        "promoted": promoted,
        "feature_importance": feature_importance,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    run_evolution()
