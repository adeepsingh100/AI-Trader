"""Audit System (Step 9, PROJECT_SPEC.md §3d) — reuse first. Everything
Step 9 asks to be traceable (timestamp/component/input/decision/output/
reason/strategy-version/confidence/market-regime/trade-id/recommendation-id)
is already a column on opportunity_evaluations, confidence_calibration, or
trades — those tables are written at every decision point in
orchestrator.run_cycle today. Rather than adding a new write path into that
hot loop (which would work against the same cycle's new per-symbol fault
isolation, per a Plan-agent pre-mortem), this module provides a READ
function joining the three into one chronological timeline, plus the two
fields Step 9 needed that weren't columns anywhere: config_version (this
file) and market_regime (added to opportunity_evaluations)."""

from __future__ import annotations

import hashlib

from src.config import (
    EXIT_SCORE_THRESHOLD,
    MIN_OPPORTUNITY_SCORE,
    OPPORTUNITY_WEIGHT_MOMENTUM,
    OPPORTUNITY_WEIGHT_RISK,
    OPPORTUNITY_WEIGHT_TREND,
    OPPORTUNITY_WEIGHT_VOLATILITY,
    OPPORTUNITY_WEIGHT_VOLUME,
    TOP_N_CANDIDATES,
)


def config_version() -> str:
    """Short, stable hash of the live scoring/threshold constants that
    determine an opportunity_evaluations row's outcome — attached to each
    row so a later reader can tell whether two evaluations ran under the
    same weights without diffing config.py by hand. Not a secret, not
    reversible-sensitive — just a fingerprint."""
    fingerprint = "|".join(
        str(v)
        for v in (
            OPPORTUNITY_WEIGHT_TREND,
            OPPORTUNITY_WEIGHT_MOMENTUM,
            OPPORTUNITY_WEIGHT_VOLUME,
            OPPORTUNITY_WEIGHT_VOLATILITY,
            OPPORTUNITY_WEIGHT_RISK,
            MIN_OPPORTUNITY_SCORE,
            EXIT_SCORE_THRESHOLD,
            TOP_N_CANDIDATES,
        )
    )
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]


def get_decision_trail(
    mode: str,
    trade_id: int | None = None,
    symbol: str | None = None,
    since=None,
) -> list[dict]:
    """Chronological, typed timeline for a trade or symbol: every
    opportunity_evaluations row in scope, each with its
    confidence_calibration row (if one was logged) nested under
    'calibration'. Read-only — no new table, no new write path."""
    from src.db import models

    client = models.get_client()
    query = client.table("opportunity_evaluations").select("*").eq("mode", mode)
    if trade_id is not None:
        query = query.eq("trade_id", trade_id)
    if symbol is not None:
        query = query.eq("symbol", symbol)
    if since is not None:
        query = query.gte("timestamp", since.isoformat())
    rows = query.order("timestamp").execute().data

    trail = []
    for row in rows:
        calibration = models.get_confidence_calibration_for_evaluation(row["id"])
        trail.append({**row, "calibration": calibration})
    return trail
