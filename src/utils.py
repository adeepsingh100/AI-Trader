"""Small, pure, stdlib-only helpers reused across features/learning/portfolio
that had drifted into independent copies in each module. Nothing here talks
to the DB or network — same "no side effects" bar as opportunity_scorer's
weighted_average(), the existing precedent for a small shared pure function."""

from __future__ import annotations

from datetime import datetime


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_timestamp(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def normalize_positive_weights(values: dict[str, float]) -> dict[str, float] | None:
    """Keeps only positive values, scaled to sum to 1.0. None if nothing
    is positive (caller falls back to its own equal-weight default)."""
    positive = {k: max(0.0, v) for k, v in values.items()}
    total = sum(positive.values())
    return {k: v / total for k, v in positive.items()} if total > 0 else None


def max_drawdown_pct(pnls: list[float], capital: float) -> float:
    """Running-peak/trough walk over an already-ordered pnl sequence,
    expressed as a % of `capital`."""
    if not pnls or capital <= 0:
        return 0.0
    running = peak = max_dd = 0.0
    for pnl in pnls:
        running += pnl
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)
    return (max_dd / capital) * 100
