"""Nightly: score the active strategy version's paper trades, ask the LLM
for an improved prompt/params (saved as a new immutable version), and
promote the version just scored to real trading if it clears the
configurable promotion bar. See PROJECT_SPEC.md §2 and §3."""

from __future__ import annotations

import json
from datetime import datetime

from src.agents.risk_manager import today_ist
from src.config import (
    PROMOTION_MAX_DRAWDOWN_PCT,
    PROMOTION_MIN_CUMULATIVE_PNL,
    PROMOTION_MIN_PAPER_DAYS,
)
from src.db import models
from src.groq_client import chat


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
    if not closed_trades or capital_to_use <= 0:
        return 0.0
    ordered = sorted(closed_trades, key=lambda t: t["closed_at"])
    running = peak = max_dd = 0.0
    for t in ordered:
        running += t["pnl"]
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)
    return (max_dd / capital_to_use) * 100


def _created_date(version: dict):
    raw = version["created_at"]
    if isinstance(raw, str):
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
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
                "and params. Respond as JSON only: "
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
    content, events = chat(messages)
    try:
        proposal = json.loads(content)
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

    models.log_agent_event(
        "evolution_agent",
        "info",
        f"metrics={metrics} promoted={promoted} new_version={new_version['version_number']}",
    )

    return {"metrics": metrics, "new_version": new_version, "promoted": promoted}


if __name__ == "__main__":
    run_evolution()
