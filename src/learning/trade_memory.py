"""Finds historical trades whose entry-time opportunity profile resembles
a current candidate — the historical evidence fed into the LLM's prompt
and, after it responds, blended into confidence calibration
(src/learning/confidence_calibration.py).

Similarity = Euclidean distance over the 5 already-computed sub-scores
(trend/momentum/volume/volatility/risk — reused from score_opportunity(),
never re-derived from raw candles). No embedded/joined Supabase query —
this codebase has no precedent for that pattern anywhere in models.py;
candidates are fetched bounded by LEARNING_HISTORY_WINDOW_DAYS/
MAX_SIMILAR_TRADES_SCANNED and matched against their trade outcome in
Python. Degrades to "no historical signal" (not a fabricated number)
whenever there's too little data or nothing close enough."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from src.config import (
    LEARNING_HISTORY_WINDOW_DAYS,
    MAX_SIMILAR_TRADES_SCANNED,
    MIN_SIMILAR_TRADES,
    SIMILARITY_MAX_DISTANCE,
    SIMILARITY_TOP_N,
)
from src.db import models
from src.learning.feature_importance import BLENDED_TIMEFRAME, SUB_SCORE_KEYS
from src.utils import normalize_positive_weights, parse_timestamp as _parse_ts

_SUB_SCORE_KEYS = ("trend_score", "momentum_score", "volume_score", "volatility_score", "risk_score")

_EMPTY_RESULT = {
    "trades": [],
    "count": 0,
    "win_rate": None,
    "avg_profit_pct": None,
    "avg_loss_pct": None,
    "avg_holding_time_seconds": None,
}


def _distance(a: dict, b: dict, weights: dict[str, float] | None = None) -> float | None:
    total, used = 0.0, 0
    for key in _SUB_SCORE_KEYS:
        av, bv = a.get(key), b.get(key)
        if av is None or bv is None:
            continue
        w = (weights or {}).get(key, 1.0)
        total += w * (av - bv) ** 2
        used += 1
    return math.sqrt(total) if used else None


def _feature_importance_weights(mode: str) -> dict[str, float] | None:
    """Reads the nightly-cached sub-score correlation weights (written by
    feature_importance.compute_subscore_correlation_weights, keyed by
    feature_name=trend_score/momentum_score/... at timeframe="blended") —
    no live recomputation, just a cheap read of a table that already
    exists. Positive correlations only, normalized to sum 1.0; None if
    nothing's cached yet or every correlation is <= 0, in which case
    _distance() falls back to its own equal-weight default."""
    rows = models.get_feature_importance(mode, timeframe=BLENDED_TIMEFRAME)
    correlations = {
        r["feature_name"]: r["correlation_score"]
        for r in rows
        if r["feature_name"] in SUB_SCORE_KEYS and r.get("correlation_score") is not None
    }
    if not correlations:
        return None
    return normalize_positive_weights(correlations)


def find_similar_trades(
    current_scores: dict,
    market_regime: str | None,
    mode: str,
    symbol: str | None = None,
    top_n: int = SIMILARITY_TOP_N,
) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    evaluations = models.get_entry_evaluations_since(mode, since)[:MAX_SIMILAR_TRADES_SCANNED]
    if not evaluations:
        return dict(_EMPTY_RESULT)

    trade_ids = [e["trade_id"] for e in evaluations if e.get("trade_id")]
    trades_by_id = {t["id"]: t for t in models.get_trades_by_ids(trade_ids)}
    weights = _feature_importance_weights(mode)

    ranked = []
    for ev in evaluations:
        trade = trades_by_id.get(ev.get("trade_id"))
        if trade is None or trade["status"] not in ("closed", "flattened") or trade["pnl"] is None:
            continue  # no known outcome yet, can't count as evidence
        if symbol is not None and trade["symbol"] != symbol:
            continue
        if market_regime is not None and trade.get("market_regime") != market_regime:
            continue
        distance = _distance(current_scores, ev, weights)
        if distance is None or distance > SIMILARITY_MAX_DISTANCE:
            continue
        ranked.append((distance, trade))

    if len(ranked) < MIN_SIMILAR_TRADES:
        return dict(_EMPTY_RESULT)

    ranked.sort(key=lambda pair: pair[0])
    top = [trade for _, trade in ranked[:top_n]]

    capital_config = models.get_capital_config(mode)
    capital_to_use = capital_config["capital_to_use"] if capital_config else None

    wins = [t["pnl"] for t in top if t["pnl"] > 0]
    losses = [t["pnl"] for t in top if t["pnl"] <= 0]
    holding_times = [
        (_parse_ts(t["closed_at"]) - _parse_ts(t["opened_at"])).total_seconds()
        for t in top
        if t.get("closed_at") and t.get("opened_at")
    ]

    return {
        "trades": top,
        "count": len(top),
        # 0-1 fraction, same convention as evolution_agent.compute_metrics —
        # callers multiply by 100 when a 0-100 confidence scale is needed.
        "win_rate": len(wins) / len(top),
        "avg_profit_pct": (sum(wins) / len(wins)) / capital_to_use * 100
        if wins and capital_to_use
        else None,
        "avg_loss_pct": (sum(losses) / len(losses)) / capital_to_use * 100
        if losses and capital_to_use
        else None,
        "avg_holding_time_seconds": sum(holding_times) / len(holding_times) if holding_times else None,
    }
