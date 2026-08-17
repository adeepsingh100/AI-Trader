"""Bucketed statistics over closed trades, plus the path-independent
"catch up on trades I haven't learned from yet" entry point.

Reuses evolution_agent.compute_metrics() for win_rate/avg_win/avg_loss/
cumulative_pnl/max_drawdown_pct rather than reimplementing that math a
second time; extends it with Sharpe/Sortino/Calmar/expectancy/
profit_factor/holding-time over a per-trade return series
(pnl / capital_to_use). Every ratio returns None on a zero/undefined
denominator rather than a fabricated number or crash — a 2-trade bucket
with no variance is "not enough data", not "Sharpe = 0.0"."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import stdev

from src.agents.evolution_agent import compute_metrics
from src.agents.risk_manager import TRADING_DAY_TZ
from src.config import (
    CONFIDENCE_ACCURACY_MIDPOINT,
    CONFIDENCE_BUCKET_WIDTH,
    LEARNING_CATCHUP_LOOKBACK_HOURS,
    LEARNING_HISTORY_WINDOW_DAYS,
    MIN_OPPORTUNITY_SCORE,
    OPPORTUNITY_SCORE_BUCKET_WIDTH,
    SORTINO_MAR_PCT,
)
from src.db import models
from src.utils import parse_timestamp as _parse_ts

_DIMENSION_TYPES = (
    "symbol",
    "market_regime",
    "opportunity_score_bucket",
    "confidence_bucket",
    "strategy_version",
    "weekday",
    "hour",
)


def _downside_deviation(returns: list[float], mar: float) -> float | None:
    if not returns:
        return None
    dd = math.sqrt(sum(min(0.0, r - mar) ** 2 for r in returns) / len(returns))
    return dd if dd else None


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


# ponytail: normal approximation, not Student's-t — valid at the sample
# floors these are actually gated behind (RECOMMENDATION_MIN_SAMPLE_SIZE,
# default 20+); revisit with a t-distribution if that floor is ever
# lowered materially below ~20.
def z_test_two_proportions(wins1: int, n1: int, wins2: int, n2: int) -> float | None:
    """Two-tailed p-value for two independent win-rate samples differing
    by chance alone. None on any degenerate input (zero samples, zero
    pooled variance) — never a fabricated p-value."""
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2 = wins1 / n1, wins2 / n2
    pooled = (wins1 + wins2) / (n1 + n2)
    variance = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if variance <= 0:
        return None
    z = (p1 - p2) / math.sqrt(variance)
    return 2 * (1 - _normal_cdf(abs(z)))


def z_test_two_means(mean1: float, stdev1: float, n1: int, mean2: float, stdev2: float, n2: int) -> float | None:
    """Two-tailed p-value for two independent sample means (e.g. expectancy
    under current vs. candidate parameters) differing by chance alone.
    Welch-style (no equal-variance assumption). None if either sample has
    fewer than 2 observations or pooled variance is 0."""
    if n1 < 2 or n2 < 2:
        return None
    variance = (stdev1**2) / n1 + (stdev2**2) / n2
    if variance <= 0:
        return None
    z = (mean1 - mean2) / math.sqrt(variance)
    return 2 * (1 - _normal_cdf(abs(z)))


def compute_bucket_statistics(trades: list[dict], capital_to_use: float) -> dict:
    closed = [t for t in trades if t.get("pnl") is not None]
    if not closed or capital_to_use <= 0:
        return {
            "trades_count": len(closed),
            "win_rate": None,
            "avg_profit": None,
            "avg_loss": None,
            "profit_factor": None,
            "expectancy": None,
            "avg_holding_time_seconds": None,
            "max_drawdown_pct": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "calmar_ratio": None,
        }

    base = compute_metrics(closed, capital_to_use)
    returns = [t["pnl"] / capital_to_use for t in closed]
    wins = [t["pnl"] for t in closed if t["pnl"] > 0]
    losses = [t["pnl"] for t in closed if t["pnl"] <= 0]
    mean_r = sum(returns) / len(returns)

    sharpe = None
    if len(returns) >= 2:
        sd = stdev(returns)
        if sd:
            sharpe = mean_r / sd

    downside_dev = _downside_deviation(returns, mar=SORTINO_MAR_PCT / 100)
    sortino = mean_r / downside_dev if downside_dev else None

    calmar = None
    if base["max_drawdown_pct"]:
        cumulative_pnl_pct = base["cumulative_pnl"] / capital_to_use * 100
        calmar = cumulative_pnl_pct / base["max_drawdown_pct"]

    sum_losses = sum(losses)
    profit_factor = sum(wins) / abs(sum_losses) if sum_losses else None

    # avg_loss is already signed negative (compute_metrics), so this nets
    # out correctly without a sign flip — don't "fix" the sign here.
    expectancy = base["win_rate"] * base["avg_win"] + (1 - base["win_rate"]) * base["avg_loss"]

    holding_times = [
        (_parse_ts(t["closed_at"]) - _parse_ts(t["opened_at"])).total_seconds()
        for t in closed
        if t.get("closed_at") and t.get("opened_at")
    ]
    avg_holding_time_seconds = sum(holding_times) / len(holding_times) if holding_times else None

    return {
        "trades_count": base["trades_count"],
        "win_rate": base["win_rate"],
        "avg_profit": base["avg_win"],
        "avg_loss": base["avg_loss"],
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_holding_time_seconds": avg_holding_time_seconds,
        "max_drawdown_pct": base["max_drawdown_pct"] or None,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
    }


def streaks(trades: list[dict]) -> dict:
    """Longest and current win/loss streaks over a chronologically-ordered
    trade list. Lives here (not reports.py, which locally-imports from
    reporting_agent to stay a leaf/reporting-only module) since the
    adaptive confidence chain's recent-performance modifier needs the
    CURRENT streak live, every cycle — making it depend on a reporting
    module would invert that layering."""
    ordered = sorted((t for t in trades if t.get("pnl") is not None), key=lambda t: t["closed_at"])
    longest_win = longest_loss = current_win = current_loss = 0
    for t in ordered:
        if t["pnl"] > 0:
            current_win, current_loss = current_win + 1, 0
        else:
            current_loss, current_win = current_loss + 1, 0
        longest_win = max(longest_win, current_win)
        longest_loss = max(longest_loss, current_loss)

    if current_win:
        current_streak_type, current_streak_length = "win", current_win
    elif current_loss:
        current_streak_type, current_streak_length = "loss", current_loss
    else:
        current_streak_type, current_streak_length = None, 0

    return {
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "current_streak_type": current_streak_type,
        "current_streak_length": current_streak_length,
    }


def _bucket_label(value: float | None, width: float) -> str | None:
    if value is None:
        return None
    lo = int(value // width * width)
    return f"{lo}-{lo + int(width)}"


def _assess_risk(trade: dict) -> str | None:
    # No configured stop-loss = unbounded downside by design, regardless
    # of outcome — a clean, real signal, not an invented threshold.
    return "too_aggressive" if trade.get("stop_loss_price") is None else "appropriate"


def _assess_stop_loss(trade: dict) -> str | None:
    if trade.get("exit_reason") != "stop_loss" or not trade.get("stop_loss_price"):
        return None
    entry, stop = trade["entry_price"], trade["stop_loss_price"]
    stop_distance_pct = abs(entry - stop) / entry * 100 if entry else None
    if stop_distance_pct is None:
        return None
    # price moved favorably by more than the stop distance before
    # reversing and stopping out — a wider stop might have let it develop
    return "too_tight" if trade.get("mfe_pct", 0) > stop_distance_pct else "appropriate"


def _assess_target(trade: dict) -> str | None:
    take_profit = trade.get("take_profit_price")
    entry = trade.get("entry_price")
    if not take_profit or not entry:
        return None
    target_distance_pct = abs(take_profit - entry) / entry * 100
    if trade.get("exit_reason") == "take_profit":
        return "realistic"
    # never hit, but price DID reach that distance via some other exit path
    return "realistic" if trade.get("mfe_pct", 0) >= target_distance_pct else "too_ambitious"


def _bucket_memberships(trade: dict, opportunity_score: float | None, confidence: float | None) -> dict:
    closed_at_ist = _parse_ts(trade["closed_at"]).astimezone(TRADING_DAY_TZ)
    memberships = {
        "symbol": trade["symbol"],
        "strategy_version": str(trade["version_id"]),
        "weekday": closed_at_ist.strftime("%A").lower(),
        "hour": str(closed_at_ist.hour),
    }
    if trade.get("market_regime"):
        memberships["market_regime"] = trade["market_regime"]
    score_bucket = _bucket_label(opportunity_score, OPPORTUNITY_SCORE_BUCKET_WIDTH)
    if score_bucket:
        memberships["opportunity_score_bucket"] = score_bucket
    confidence_bucket = _bucket_label(confidence, CONFIDENCE_BUCKET_WIDTH)
    if confidence_bucket:
        memberships["confidence_bucket"] = confidence_bucket
    return memberships


def _evaluate_trade(trade: dict) -> dict:
    won = trade["pnl"] is not None and trade["pnl"] > 0
    entry_eval = models.get_entry_evaluation_for_trade(trade["id"])
    predicted_opportunity_score = entry_eval["opportunity_score"] if entry_eval else None
    calibration = (
        models.get_confidence_calibration_for_evaluation(entry_eval["id"]) if entry_eval else None
    )
    predicted_confidence = calibration["final_confidence"] if calibration else None

    confidence_was_accurate = None
    if predicted_confidence is not None:
        confidence_was_accurate = (predicted_confidence >= CONFIDENCE_ACCURACY_MIDPOINT) == won

    opportunity_score_was_accurate = None
    if predicted_opportunity_score is not None:
        opportunity_score_was_accurate = (predicted_opportunity_score >= MIN_OPPORTUNITY_SCORE) == won

    models.upsert_trade_evaluation(
        trade_id=trade["id"],
        predicted_confidence=predicted_confidence,
        predicted_opportunity_score=predicted_opportunity_score,
        actual_outcome_won=won,
        confidence_was_accurate=confidence_was_accurate,
        opportunity_score_was_accurate=opportunity_score_was_accurate,
        risk_assessment=_assess_risk(trade),
        stop_loss_assessment=_assess_stop_loss(trade),
        target_assessment=_assess_target(trade),
    )
    return {"opportunity_score": predicted_opportunity_score, "confidence": predicted_confidence}


def _update_statistics_buckets(mode: str, bucket_keys: set[tuple[str, str]], capital_to_use: float) -> None:
    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    for dimension_type, dimension_value in bucket_keys:
        all_recent = models.get_recently_closed_trades(mode, since)
        if dimension_type == "symbol":
            bucket_trades = [t for t in all_recent if t["symbol"] == dimension_value]
        elif dimension_type == "market_regime":
            bucket_trades = [t for t in all_recent if t.get("market_regime") == dimension_value]
        elif dimension_type == "strategy_version":
            bucket_trades = [t for t in all_recent if str(t["version_id"]) == dimension_value]
        elif dimension_type in ("weekday", "hour"):
            bucket_trades = [
                t
                for t in all_recent
                if _bucket_memberships(t, None, None).get(dimension_type) == dimension_value
            ]
        else:
            # opportunity_score_bucket / confidence_bucket need each trade's
            # predicted values, which aren't on the trade row itself.
            bucket_trades = []
            for t in all_recent:
                entry_eval = models.get_entry_evaluation_for_trade(t["id"])
                score = entry_eval["opportunity_score"] if entry_eval else None
                confidence = None
                if entry_eval:
                    calibration = models.get_confidence_calibration_for_evaluation(entry_eval["id"])
                    confidence = calibration["final_confidence"] if calibration else None
                label = (
                    _bucket_label(score, OPPORTUNITY_SCORE_BUCKET_WIDTH)
                    if dimension_type == "opportunity_score_bucket"
                    else _bucket_label(confidence, CONFIDENCE_BUCKET_WIDTH)
                )
                if label == dimension_value:
                    bucket_trades.append(t)

        stats = compute_bucket_statistics(bucket_trades, capital_to_use)
        models.upsert_learning_statistics(mode, dimension_type, dimension_value, stats)


def process_closed_trades(mode: str) -> list[dict]:
    """Path-independent catch-up: finds closed trades not yet
    self-evaluated (regardless of whether they closed via the SL/TP
    sweep, an LLM-validated exit, or a circuit-breaker flatten) and
    processes exactly those. Called once at the end of run_cycle()."""
    since = datetime.now(timezone.utc) - timedelta(hours=LEARNING_CATCHUP_LOOKBACK_HOURS)
    recent = models.get_recently_closed_trades(mode, since)
    if not recent:
        return []

    already_evaluated = models.get_trade_evaluation_ids([t["id"] for t in recent])
    new_trades = [t for t in recent if t["id"] not in already_evaluated]
    if not new_trades:
        return []

    capital_config = models.get_capital_config(mode)
    capital_to_use = capital_config["capital_to_use"] if capital_config else 0

    touched_buckets: set[tuple[str, str]] = set()
    for trade in new_trades:
        predicted = _evaluate_trade(trade)
        memberships = _bucket_memberships(
            trade, predicted["opportunity_score"], predicted["confidence"]
        )
        touched_buckets.update(memberships.items())

    _update_statistics_buckets(mode, touched_buckets, capital_to_use)
    return new_trades
