"""Root Cause Analysis (Step 2, Scientific Strategy Optimization
Framework). orchestrator.py already logs a reason + risk_manager_result on
every scanned symbol every cycle, including candidates that never became a
trade (final_decision="hold") — fully captured, previously never read back
by anything. This ranks WHY, replacing "no trades" with a breakdown."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import LEARNING_HISTORY_WINDOW_DAYS
from src.db import models


def _rejection_label(row: dict) -> str:
    """risk_manager_result, when present, is the more specific reason (e.g.
    "block_concentration_limit") — reason alone (e.g. "not_a_candidate",
    "confidence gated: ...") otherwise."""
    return row.get("risk_manager_result") or row.get("reason") or "unknown"


def rejection_breakdown(mode: str, since: datetime | None = None) -> list[dict]:
    """Ranked [{"reason", "count", "pct_of_rejections"}, ...], descending
    by count — the "Volume Filter 38%, Momentum Filter 24%..." ask."""
    since = since or (datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS))
    rows = models.get_hold_evaluations_since(mode, since)
    if not rows:
        return []

    counts: dict[str, int] = {}
    for row in rows:
        label = _rejection_label(row)
        counts[label] = counts.get(label, 0) + 1

    total = len(rows)
    breakdown = [
        {"reason": reason, "count": count, "pct_of_rejections": count / total * 100}
        for reason, count in counts.items()
    ]
    breakdown.sort(key=lambda r: r["count"], reverse=True)
    return breakdown
